# src/exercises/squat.py
from typing import Optional, Dict, Tuple, List
import time
from analysis.metrics import stage_update

# ROM semantics from frame_metrics:
#   0%   = top/standing tall
#   100% = bottom/deepest knee flexion we expect

CFG = {
    # Stage machine thresholds on ROM%
    "down_enter": 65.0,   # enter DOWN when ROM >= 65%
    "down_exit":  50.0,   # stay DOWN while ROM >= 50%
    "up_enter":   10.0,   # enter UP when ROM <= 10%
    "up_exit":    20.0,   # stay UP while ROM <= 20%
    "hold_ms":   150,
    "min_speed":  6.0,    # ROM%/s

    # Depth tiers (peak ROM during the attempt)
    "quarter_depth_pct":  35.0,
    "half_depth_pct":     60.0,
    "depth_target_pct":   85.0,   # stricter? bump to 90 once tuned

    # Lockout requirement at finish (top ROM small)
    "lockout_target_pct": 12.0,

    # Form thresholds
    "torso_tilt_warn_deg":  45.0,   # >45° is a warning (bodyweight)
    "valgus_warn_abs":       0.030, # normalized x-offset (~3% of width)
    "midfoot_mismatch":      0.060, # |shoulder_mid_x - midfoot_x| over this → warning
    "bounce_speed_warn":     110.0, # if |vel| spikes past this near reversal, warn
}

class SquatDetector:
    """
    Bodyweight squat detector with partial-rep filtering and form warnings.

    Expected fields in fm (frame_metrics):
      - rom_squat_smooth, vel_squat
      - r_knee_deg, l_knee_deg (not strictly required for counting)
      - torso_tilt_deg
      - knee_valgus_r, knee_valgus_l
      - shoulder_mid_x, midfoot_x

    Contract:
      - update(fm, now_s) → (attempt_event|None, live_state)
      - attempt_event fires on DOWN -> UP.
      - Only increments rep_count when depth and lockout are satisfied.
    """
    def __init__(self, cfg: Dict = None):
        self.cfg = {**CFG, **(cfg or {})}

        self.stage: Optional[str] = None
        self.db: Optional[Dict]   = None

        self.rep_count: int = 0          # valid reps
        self.attempt_count: int = 0      # all DOWN->UP cycles

        # per-attempt accumulators
        self._rom_min = None
        self._rom_max = None
        self._tilt_peak = None
        self._valgus_max_abs = 0.0
        self._midfoot_delta_max = 0.0
        self._min_neg_vel = 0.0
        self._max_pos_vel = 0.0

        # tempo
        self._phase_start_s: Optional[float] = None
        self._time_down_s = 0.0
        self._time_up_s = 0.0

    # ---------- accumulation ----------
    def _accumulate(self, fm: Dict):
        rom = fm.get("rom_squat_smooth")
        if rom is not None:
            self._rom_min = rom if self._rom_min is None else min(self._rom_min, rom)
            self._rom_max = rom if self._rom_max is None else max(self._rom_max, rom)

        tilt = fm.get("torso_tilt_deg")
        if tilt is not None:
            self._tilt_peak = tilt if self._tilt_peak is None else max(self._tilt_peak, tilt)

        # valgus: track worst absolute offset of either knee
        kv_r = fm.get("knee_valgus_r")
        kv_l = fm.get("knee_valgus_l")
        for kv in (kv_r, kv_l):
            if kv is not None:
                self._valgus_max_abs = max(self._valgus_max_abs, abs(kv))

        # balance: shoulder over midfoot proxy
        smx = fm.get("shoulder_mid_x")
        mfx = fm.get("midfoot_x")
        if smx is not None and mfx is not None:
            self._midfoot_delta_max = max(self._midfoot_delta_max, abs(smx - mfx))

        # bounce: track velocity extremes
        v = fm.get("vel_squat")
        if isinstance(v, (int, float)):
            self._min_neg_vel = min(self._min_neg_vel, v)  # most negative
            self._max_pos_vel = max(self._max_pos_vel, v)  # most positive

    def _reset_accumulators(self):
        self._rom_min = self._rom_max = None
        self._tilt_peak = None
        self._valgus_max_abs = 0.0
        self._midfoot_delta_max = 0.0
        self._min_neg_vel = 0.0
        self._max_pos_vel = 0.0
        self._time_down_s = 0.0
        self._time_up_s = 0.0

    # ---------- classification ----------
    def _classify_attempt(self) -> tuple[str, List[str], bool]:
        """
        Returns (rep_class, cues, should_count).
        """
        cues: List[str] = []
        cls = "good"
        count_valid = True

        depth_peak = self._rom_max if self._rom_max is not None else -1.0
        top_rom    = self._rom_min if self._rom_min is not None else 1e9

        # depth tiers
        if depth_peak < self.cfg["quarter_depth_pct"]:
            cls, count_valid = "quarter_rep", False
            cues.append("Go MUCH deeper")
        elif depth_peak < self.cfg["half_depth_pct"]:
            cls, count_valid = "half_rep", False
            cues.append("Go deeper (past halfway)")
        elif depth_peak < self.cfg["depth_target_pct"]:
            cls, count_valid = "depth_short", False
            cues.append(f"Hit ≥ {int(self.cfg['depth_target_pct'])}% depth")

        # lockout only matters if depth was okay
        lockout_ok = top_rom <= self.cfg["lockout_target_pct"]
        if cls == "good" and not lockout_ok:
            cls, count_valid = "lockout_fail", False
            cues.append("Stand tall—finish lockout")

        # posture warnings (non-fatal)
        if self._tilt_peak is not None and self._tilt_peak > self.cfg["torso_tilt_warn_deg"]:
            if cls == "good": cls = "form_warn"
            cues.append("Keep chest up—reduce torso tilt")

        if self._valgus_max_abs > self.cfg["valgus_warn_abs"]:
            if cls == "good": cls = "form_warn"
            cues.append("Knees out—avoid valgus")

        if self._midfoot_delta_max > self.cfg["midfoot_mismatch"]:
            if cls == "good": cls = "form_warn"
            cues.append("Stay over midfoot")

        # bounce warning: very fast reversal
        if abs(self._min_neg_vel) > self.cfg["bounce_speed_warn"] and self._max_pos_vel > self.cfg["bounce_speed_warn"]:
            if cls == "good": cls = "form_warn"
            cues.append("Control the bottom—no bounce")

        return cls, cues[:2], count_valid

    # ---------- public ----------
    def update(self, fm: Dict[str, Optional[float]], now_s: Optional[float] = None):
        if now_s is None:
            now_s = time.time()

        rom = fm.get("rom_squat_smooth")
        vel = fm.get("vel_squat")

        self._accumulate(fm)

        prev_stage = self.stage
        self.stage, self.db = stage_update(
            prev_stage=prev_stage, rom=rom, vel=vel,
            now_s=now_s, cfg=self.cfg, db=self.db
        )

        # tempo
        if self._phase_start_s is None:
            self._phase_start_s = now_s
        else:
            dt = max(0.0, now_s - self._phase_start_s)
            if prev_stage == "DOWN":
                self._time_down_s += dt
            elif prev_stage == "UP":
                self._time_up_s += dt
            self._phase_start_s = now_s

        attempt_event = None

        # completed attempt = DOWN -> UP
        if prev_stage == "DOWN" and self.stage == "UP":
            self.attempt_count += 1
            rep_class, cues, should_count = self._classify_attempt()
            if should_count:
                self.rep_count += 1

            attempt_event = {
                "ts": now_s,
                "exercise": "squat",
                "attempt_index": self.attempt_count,
                "rep_index_valid": self.rep_count if should_count else None,
                "counted": should_count,
                "class": rep_class,
                "cues": cues,
                "snapshot": {
                    "rom_peak_pct": self._rom_max,
                    "rom_top_pct": self._rom_min,
                    "torso_tilt_peak_deg": self._tilt_peak,
                    "valgus_max_abs": self._valgus_max_abs,
                    "midfoot_delta_max": self._midfoot_delta_max,
                    "min_neg_vel": self._min_neg_vel,
                    "max_pos_vel": self._max_pos_vel,
                    "tempo_down_s": round(self._time_down_s, 3),
                    "tempo_up_s": round(self._time_up_s, 3),
                },
            }
            self._reset_accumulators()

        live = {
            "stage": self.stage,
            "rep_count": self.rep_count,
            "attempt_count": self.attempt_count,
            "rom": rom,
            "vel": vel,
        }
        return attempt_event, live

__all__ = ["SquatDetector", "CFG"]
