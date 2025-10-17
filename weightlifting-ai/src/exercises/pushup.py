from typing import Optional, Dict, Tuple, List
import time
from analysis.metrics import stage_update

# -----------------------------
# ROM conventions (already set upstream in frame_metrics):
#   0%   = top/lockout (straight arms)
#   100% = bottom/deepest expected elbow bend
# We only COUNT a rep when BOTH:
#   - depth_peak >= depth_target_pct
#   - final top ROM <= lockout_target_pct
# Everything else becomes a classified attempt (quarter/half/depth_short/lockout_fail)
# -----------------------------

CFG = {
    # Stage machine hysteresis on ROM%
    "down_enter": 70.0,
    "down_exit":  55.0,
    "up_enter":   10.0,
    "up_exit":    20.0,

    # Debounce and speed gate
    "hold_ms": 150,
    "min_speed": 8.0,     # ROM %/s minimum magnitude to allow switching

    # Depth classification thresholds (peak ROM during the down phase)
    "quarter_depth_pct":   35.0,  # <35% → "quarter_rep"
    "half_depth_pct":      60.0,  # <60% → "half_rep"
    "depth_target_pct":    90.0,  # >=90% required to be eligible as a valid rep

    # Lockout requirement (ROM at finish should be near 0%)
    "lockout_target_pct":  12.0,  # <=12% at finish counts as locked out

    # Posture/quality angles (optional, informational)
    "torso_tilt_max_deg":  35.0,  # warn if exceeded at any point
    "hip_sag_min_deg":    160.0,  # warn if hip angle min < this
    "lockout_min_elbow_deg": 165.0, # backup check using elbow angle at top (optional)
}

class PushUpDetector:
    """
    Push-up detector with strict depth/lockout handling and partial rep awareness.

    update(fm, now_s) -> (rep_event|None, live_state)
      - fm: dict from analysis.frame_metrics.compute_frame_metrics
        keys used:
          rom_pushup_smooth, vel_pushup,
          r_elbow_deg, l_elbow_deg,
          torso_tilt_deg, r_hip_deg, l_hip_deg
      - On DOWN -> UP transition, emits a rep_event with class and cues.
      - Only increments rep_count when depth AND lockout requirements are met.
        attempt_count increments on every DOWN -> UP regardless.
    """
    def __init__(self, cfg: Dict = None):
        self.cfg = {**CFG, **(cfg or {})}

        self.stage: Optional[str] = None
        self.db: Optional[Dict] = None

        # Counters
        self.rep_count: int = 0          # valid reps
        self.attempt_count: int = 0      # all DOWN->UP cycles

        # Per-attempt accumulators (reset on each completed attempt)
        self._rom_min = None             # lowest ROM seen (near top)
        self._rom_max = None             # highest ROM seen (near bottom)
        self._tilt_peak = None           # peak torso tilt
        self._elbow_top = None           # largest elbow angle at top (lockout quality)
        self._hip_min = None             # smallest hip angle (sag)

        # Tempo timing (optional)
        self._phase_start_s: Optional[float] = None
        self._time_down_s: float = 0.0
        self._time_up_s: float = 0.0

    # ---------- internal accumulation ----------
    def _accumulate(self, fm: Dict):
        # ROM extrema
        rom = fm.get("rom_pushup_smooth")
        if rom is not None:
            self._rom_min = rom if self._rom_min is None else min(self._rom_min, rom)
            self._rom_max = rom if self._rom_max is None else max(self._rom_max, rom)

        # Top lockout quality via elbow angle
        elbows = [fm.get("r_elbow_deg"), fm.get("l_elbow_deg")]
        elbows = [x for x in elbows if x is not None]
        if elbows:
            e_max = max(elbows)
            self._elbow_top = e_max if self._elbow_top is None else max(self._elbow_top, e_max)

        # Torso tilt
        tilt = fm.get("torso_tilt_deg")
        if tilt is not None:
            self._tilt_peak = tilt if self._tilt_peak is None else max(self._tilt_peak, tilt)

        # Hip sag (use min of sides)
        hips = [fm.get("r_hip_deg"), fm.get("l_hip_deg")]
        hips = [x for x in hips if x is not None]
        if hips:
            h_min = min(hips)
            self._hip_min = h_min if self._hip_min is None else min(self._hip_min, h_min)

    def _reset_accumulators(self):
        self._rom_min = self._rom_max = None
        self._tilt_peak = None
        self._elbow_top = None
        self._hip_min = None
        self._time_down_s = 0.0
        self._time_up_s = 0.0

    # ---------- classification helpers ----------
    def _classify_attempt(self) -> Tuple[str, List[str], bool]:
        """
        Returns (rep_class, cues, count_valid)
          - rep_class: 'good' | 'quarter_rep' | 'half_rep' | 'depth_short' | 'lockout_fail' | 'form_warn'
          - cues: up to 2 strings
          - count_valid: True only if it should increment rep_count
        """
        cues: List[str] = []
        cls = "good"
        count_valid = True

        depth_peak = self._rom_max if self._rom_max is not None else -1.0
        top_rom   = self._rom_min if self._rom_min is not None else 1e9

        # Depth tiers
        if depth_peak < self.cfg["quarter_depth_pct"]:
            cls, count_valid = "quarter_rep", False
            cues.append("Go MUCH deeper")
        elif depth_peak < self.cfg["half_depth_pct"]:
            cls, count_valid = "half_rep", False
            cues.append("Go deeper (past halfway)")
        elif depth_peak < self.cfg["depth_target_pct"]:
            cls, count_valid = "depth_short", False
            cues.append(f"Depth target ≥ {int(self.cfg['depth_target_pct'])}% ROM")

        # Lockout check (only matters if depth is okay)
        lockout_ok = top_rom <= self.cfg["lockout_target_pct"]
        if cls == "good" and not lockout_ok:
            cls, count_valid = "lockout_fail", False
            cues.append("Finish lockout at the top")

        # Posture warnings (non-fatal)
        if self._tilt_peak is not None and self._tilt_peak > self.cfg["torso_tilt_max_deg"]:
            if cls == "good": cls = "form_warn"
            cues.append("Stay tighter—reduce torso tilt")
        if self._hip_min is not None and self._hip_min < self.cfg["hip_sag_min_deg"]:
            if cls == "good": cls = "form_warn"
            cues.append("Avoid hip sag—brace core & glutes")

        return cls, cues[:2], count_valid

    # ---------- public ----------
    def update(self, fm: Dict[str, Optional[float]], now_s: Optional[float] = None):
        """
        Feed one frame's metrics. Returns (rep_event_or_None, live_state).
          - live_state: {"stage", "rep_count", "attempt_count", "rom", "vel"}
          - rep_event has: class, cues, and per-attempt snapshot.
        """
        if now_s is None:
            now_s = time.time()

        rom = fm.get("rom_pushup_smooth")
        vel = fm.get("vel_pushup")

        self._accumulate(fm)

        prev_stage = self.stage
        self.stage, self.db = stage_update(
            prev_stage=prev_stage, rom=rom, vel=vel,
            now_s=now_s, cfg=self.cfg, db=self.db
        )

        # Tempo accumulation
        if self._phase_start_s is None:
            self._phase_start_s = now_s
        else:
            dt = max(0.0, now_s - self._phase_start_s)
            if prev_stage == "DOWN":
                self._time_down_s += dt
            elif prev_stage == "UP":
                self._time_up_s += dt
            self._phase_start_s = now_s

        rep_event = None

        # A completed attempt is strictly the transition DOWN -> UP
        if prev_stage == "DOWN" and self.stage == "UP":
            self.attempt_count += 1
            rep_class, cues, count_valid = self._classify_attempt()

            if count_valid:
                self.rep_count += 1

            rep_event = {
                "ts": now_s,
                "exercise": "pushup",
                "rep_index_valid": self.rep_count if count_valid else None,
                "attempt_index": self.attempt_count,
                "counted": count_valid,
                "class": rep_class,
                "cues": cues,
                "snapshot": {
                    "rom_peak_pct": self._rom_max,
                    "rom_top_pct": self._rom_min,
                    "torso_tilt_peak_deg": self._tilt_peak,
                    "elbow_top_deg": self._elbow_top,
                    "hip_angle_min_deg": self._hip_min,
                    "tempo_down_s": round(self._time_down_s, 3),
                    "tempo_up_s": round(self._time_up_s, 3),
                },
            }

            # Reset per-attempt state
            self._reset_accumulators()

        live_state = {
            "stage": self.stage,
            "rep_count": self.rep_count,
            "attempt_count": self.attempt_count,
            "rom": rom,
            "vel": vel,
        }
        return rep_event, live_state

__all__ = ["PushUpDetector", "CFG"]
