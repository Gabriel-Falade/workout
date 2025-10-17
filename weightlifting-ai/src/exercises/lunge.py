# src/exercises/lunge.py
from typing import Optional, Dict, List, Tuple
import time
from analysis.metrics import rom_percent, ema, velocity
from analysis.metrics import stage_update

# ROM definition:
#   0%   = top/lockout (knee almost straight)
#   100% = bottom/deepest front-knee flexion
CFG = {
    # Knee angle refs (deg) for ROM mapping
    "top_deg": 175.0,
    "bottom_deg": 70.0,

    # Stage machine thresholds on ROM%
    "down_enter": 65.0,
    "down_exit":  50.0,
    "up_enter":   10.0,
    "up_exit":    20.0,
    "hold_ms":   150,
    "min_speed":  6.0,    # ROM%/s (gate stage flipping)

    # Depth tiers (peak ROM during attempt)
    "quarter_depth_pct":  30.0,
    "half_depth_pct":     55.0,
    "depth_target_pct":   85.0,

    # Lockout requirement at finish
    "lockout_target_pct": 12.0,

    # Posture/balance warnings
    "valgus_warn_abs":     0.030,  # medial drift proxy (abs)
    "torso_tilt_warn_deg": 40.0,
    "midfoot_mismatch":    0.060,  # |shoulder_mid_x - midfoot_x|
    "bounce_speed_warn":  110.0,   # large decel/accel around reversal

    # EMA smoothing
    "ema_alpha_rom": 0.30,
}

class LungeDetector:
    """
    Front-lunge detector. Auto-picks the active (front) leg each frame by
    whichever knee has higher ROM% (more flexion). Counts only if depth and
    lockout are satisfied. Tracks attempts vs valid reps.
    """
    def __init__(self, cfg: Dict = None):
        self.cfg = {**CFG, **(cfg or {})}
        self.stage: Optional[str] = None
        self.db: Optional[Dict] = None

        self.rep_count = 0
        self.attempt_count = 0

        # live smoothing (internal, independent of frame_metrics)
        self._rom_smooth: Optional[float] = None
        self._last_rom_smooth: Optional[float] = None
        self._last_ts: Optional[float] = None
        self._vel: Optional[float] = None

        # per-attempt accumulators
        self._rom_min = None
        self._rom_max = None
        self._tilt_peak = None
        self._valgus_max_abs = 0.0
        self._midfoot_delta_max = 0.0
        self._min_neg_vel = 0.0
        self._max_pos_vel = 0.0
        self._active_side_seen = None  # "left"/"right" (front leg heuristic)

        # tempo timing
        self._phase_start_s: Optional[float] = None
        self._time_down_s = 0.0
        self._time_up_s = 0.0

    # --- helpers ---
    def _compute_rom_pair(self, fm: Dict) -> Tuple[Optional[float], Optional[str]]:
        """Return (rom%, active_side) using knee angles and ROM mapping."""
        lk = fm.get("l_knee_deg")
        rk = fm.get("r_knee_deg")
        top = self.cfg["top_deg"]; bot = self.cfg["bottom_deg"]

        rl = rom_percent(lk, top, bot) if lk is not None else None
        rr = rom_percent(rk, top, bot) if rk is not None else None

        # pick side with larger ROM (more flexion) as active
        if rl is None and rr is None:
            return None, None
        if rl is None:   return rr, "right"
        if rr is None:   return rl, "left"
        if rr >= rl:     return rr, "right"
        else:            return rl, "left"

    def _accumulate(self, fm: Dict, rom: Optional[float], active_side: Optional[str]):
        if rom is not None:
            self._rom_min = rom if self._rom_min is None else min(self._rom_min, rom)
            self._rom_max = rom if self._rom_max is None else max(self._rom_max, rom)

        self._active_side_seen = active_side or self._active_side_seen

        tilt = fm.get("torso_tilt_deg")
        if tilt is not None:
            self._tilt_peak = tilt if self._tilt_peak is None else max(self._tilt_peak, tilt)

        # valgus (track worst abs drift for either knee)
        for kv in (fm.get("knee_valgus_l"), fm.get("knee_valgus_r")):
            if kv is not None:
                self._valgus_max_abs = max(self._valgus_max_abs, abs(kv))

        # balance: shoulder over midfoot
        smx = fm.get("shoulder_mid_x"); mfx = fm.get("midfoot_x")
        if smx is not None and mfx is not None:
            self._midfoot_delta_max = max(self._midfoot_delta_max, abs(smx - mfx))

        # velocity extremes (for bounce warning)
        if isinstance(self._vel, (int, float)):
            self._min_neg_vel = min(self._min_neg_vel, self._vel)
            self._max_pos_vel = max(self._max_pos_vel, self._vel)

    def _reset_attempt(self):
        self._rom_min = self._rom_max = None
        self._tilt_peak = None
        self._valgus_max_abs = 0.0
        self._midfoot_delta_max = 0.0
        self._min_neg_vel = 0.0
        self._max_pos_vel = 0.0
        self._active_side_seen = None
        self._time_down_s = 0.0
        self._time_up_s = 0.0

    def _classify(self) -> Tuple[str, List[str], bool]:
        cues: List[str] = []
        cls = "good"; count_valid = True

        depth_peak = self._rom_max if self._rom_max is not None else -1.0
        top_rom    = self._rom_min if self._rom_min is not None else 1e9

        # depth gating/tiers
        if depth_peak < self.cfg["quarter_depth_pct"]:
            cls, count_valid = "quarter_rep", False; cues.append("Go MUCH deeper")
        elif depth_peak < self.cfg["half_depth_pct"]:
            cls, count_valid = "half_rep", False; cues.append("Go deeper (past halfway)")
        elif depth_peak < self.cfg["depth_target_pct"]:
            cls, count_valid = "depth_short", False; cues.append(f"Hit ≥ {int(self.cfg['depth_target_pct'])}% depth")

        # lockout only if depth ok
        lock_ok = top_rom <= self.cfg["lockout_target_pct"]
        if cls == "good" and not lock_ok:
            cls, count_valid = "lockout_fail", False; cues.append("Stand tall—finish lockout")

        # warnings (non-fatal)
        if self._tilt_peak is not None and self._tilt_peak > self.cfg["torso_tilt_warn_deg"]:
            if cls == "good": cls = "form_warn"; cues.append("Keep torso upright")
        if self._valgus_max_abs > self.cfg["valgus_warn_abs"]:
            if cls == "good": cls = "form_warn"; cues.append("Knee tracking—avoid valgus")
        if self._midfoot_delta_max > self.cfg["midfoot_mismatch"]:
            if cls == "good": cls = "form_warn"; cues.append("Balance over midfoot")
        if abs(self._min_neg_vel) > self.cfg["bounce_speed_warn"] and self._max_pos_vel > self.cfg["bounce_speed_warn"]:
            if cls == "good": cls = "form_warn"; cues.append("Control the bottom—no bounce")

        return cls, cues[:2], count_valid

    # --- public ---
    def update(self, fm: Dict[str, Optional[float]], now_s: Optional[float] = None):
        if now_s is None:
            now_s = time.time()

        # dt for our internal smoothing/vel
        if self._last_ts is None:
            dt = 1/60.0
        else:
            dt = max(1e-3, now_s - self._last_ts)
        self._last_ts = now_s

        # ROM calc from knee angles
        rom, active_side = self._compute_rom_pair(fm)
        self._rom_smooth = ema(self._rom_smooth, rom, self.cfg["ema_alpha_rom"]) if rom is not None else self._rom_smooth
        self._vel = velocity(self._rom_smooth, self._last_rom_smooth, dt) if self._rom_smooth is not None else None
        self._last_rom_smooth = self._rom_smooth if self._rom_smooth is not None else self._last_rom_smooth

        # accumulate per-attempt info
        self._accumulate(fm, self._rom_smooth if self._rom_smooth is not None else rom, active_side)

        # stage machine on smoothed ROM
        prev_stage = self.stage
        self.stage, self.db = stage_update(
            prev_stage=prev_stage, rom=self._rom_smooth, vel=self._vel,
            now_s=now_s, cfg=self.cfg, db=self.db
        )

        # tempo timing
        if self._phase_start_s is None:
            self._phase_start_s = now_s
        else:
            dti = max(0.0, now_s - self._phase_start_s)
            if prev_stage == "DOWN": self._time_down_s += dti
            elif prev_stage == "UP": self._time_up_s += dti
            self._phase_start_s = now_s

        attempt_event = None
        if prev_stage == "DOWN" and self.stage == "UP":
            self.attempt_count += 1
            rep_class, cues, counted = self._classify()
            if counted:
                self.rep_count += 1

            attempt_event = {
                "ts": now_s,
                "exercise": "lunge",
                "attempt_index": self.attempt_count,
                "rep_index_valid": self.rep_count if counted else None,
                "counted": counted,
                "class": rep_class,
                "cues": cues,
                "snapshot": {
                    "rom_peak_pct": self._rom_max,
                    "rom_top_pct": self._rom_min,
                    "active_side": self._active_side_seen,
                    "torso_tilt_peak_deg": self._tilt_peak,
                    "valgus_max_abs": self._valgus_max_abs,
                    "midfoot_delta_max": self._midfoot_delta_max,
                    "tempo_down_s": round(self._time_down_s, 3),
                    "tempo_up_s": round(self._time_up_s, 3),
                }
            }
            self._reset_attempt()

        live = {
            "stage": self.stage,
            "rep_count": self.rep_count,
            "attempt_count": self.attempt_count,
            "rom": self._rom_smooth if self._rom_smooth is not None else rom,
            "vel": self._vel,
        }
        return attempt_event, live

__all__ = ["LungeDetector", "CFG"]
