# src/exercises/situp.py
from typing import Optional, Dict, List, Tuple
import time
from analysis.metrics import rom_percent, ema, velocity, stage_update

# ROM definition using hip flexion:
#   0%   = lying down (hip ~straight, ~170-180°)
#   100% = fully up (hip flexed, ~60°)
CFG = {
    "top_deg": 170.0,
    "bottom_deg": 60.0,

    "down_enter": 65.0,   # "down" = going into the sit-up (toward 100%)
    "down_exit":  50.0,
    "up_enter":   10.0,   # "up" = back to lying down (toward 0%)
    "up_exit":    20.0,
    "hold_ms":   120,
    "min_speed":  6.0,

    # depth tiers (peak ROM during attempt)
    "quarter_depth_pct":  25.0,
    "half_depth_pct":     50.0,
    "depth_target_pct":   85.0,

    # lockout at finish (back to near 0% ROM)
    "lockout_target_pct": 12.0,

    # warnings
    "torso_tilt_warn_deg": 55.0,  # if camera is not side-on, relax this
    "bounce_speed_warn": 120.0,

    "ema_alpha_rom": 0.30,
}

class SitupDetector:
    """
    Sit-up detector using hip angles (shoulder-hip-knee).
    Assumes roughly side-on camera for best signal.
    Counts only when depth (peak ROM) and final lockout are satisfied.
    """
    def __init__(self, cfg: Dict = None):
        self.cfg = {**CFG, **(cfg or {})}
        self.stage: Optional[str] = None
        self.db: Optional[Dict] = None

        self.rep_count = 0
        self.attempt_count = 0

        self._rom_smooth: Optional[float] = None
        self._last_rom_smooth: Optional[float] = None
        self._last_ts: Optional[float] = None
        self._vel: Optional[float] = None

        # per-attempt accumulators
        self._rom_min = None
        self._rom_max = None
        self._tilt_peak = None
        self._min_neg_vel = 0.0
        self._max_pos_vel = 0.0

        # tempo
        self._phase_start_s: Optional[float] = None
        self._time_down_s = 0.0
        self._time_up_s = 0.0

    def _hip_angle_best(self, fm: Dict) -> Optional[float]:
        """Use the smaller hip angle (more flexion) across sides."""
        lh = fm.get("l_hip_deg"); rh = fm.get("r_hip_deg")
        if lh is None and rh is None: return None
        if lh is None: return rh
        if rh is None: return lh
        return min(lh, rh)

    def _accumulate(self, fm: Dict, rom: Optional[float]):
        if rom is not None:
            self._rom_min = rom if self._rom_min is None else min(self._rom_min, rom)
            self._rom_max = rom if self._rom_max is None else max(self._rom_max, rom)

        tilt = fm.get("torso_tilt_deg")
        if tilt is not None:
            self._tilt_peak = tilt if self._tilt_peak is None else max(self._tilt_peak, tilt)

        if isinstance(self._vel, (int, float)):
            self._min_neg_vel = min(self._min_neg_vel, self._vel)
            self._max_pos_vel = max(self._max_pos_vel, self._vel)

    def _reset_attempt(self):
        self._rom_min = self._rom_max = None
        self._tilt_peak = None
        self._min_neg_vel = 0.0
        self._max_pos_vel = 0.0
        self._time_down_s = 0.0
        self._time_up_s = 0.0

    def _classify(self) -> Tuple[str, List[str], bool]:
        cues: List[str] = []
        cls = "good"; count_valid = True

        depth_peak = self._rom_max if self._rom_max is not None else -1.0
        top_rom    = self._rom_min if self._rom_min is not None else 1e9

        if depth_peak < self.cfg["quarter_depth_pct"]:
            cls, count_valid = "quarter_rep", False; cues.append("Go MUCH higher")
        elif depth_peak < self.cfg["half_depth_pct"]:
            cls, count_valid = "half_rep", False; cues.append("Go higher (past halfway)")
        elif depth_peak < self.cfg["depth_target_pct"]:
            cls, count_valid = "depth_short", False; cues.append(f"Hit ≥ {int(self.cfg['depth_target_pct'])}% up")

        lock_ok = top_rom <= self.cfg["lockout_target_pct"]
        if cls == "good" and not lock_ok:
            cls, count_valid = "lockout_fail", False; cues.append("Return fully to the floor")

        if self._tilt_peak is not None and self._tilt_peak > self.cfg["torso_tilt_warn_deg"]:
            if cls == "good": cls = "form_warn"; cues.append("Control torso tilt")
        if abs(self._min_neg_vel) > self.cfg["bounce_speed_warn"] and self._max_pos_vel > self.cfg["bounce_speed_warn"]:
            if cls == "good": cls = "form_warn"; cues.append("Control—no bounce")

        return cls, cues[:2], count_valid

    def update(self, fm: Dict[str, Optional[float]], now_s: Optional[float] = None):
        if now_s is None:
            now_s = time.time()

        # dt for smoothing/vel
        if self._last_ts is None: dt = 1/60.0
        else: dt = max(1e-3, now_s - self._last_ts)
        self._last_ts = now_s

        # ROM from hip angle
        hip = self._hip_angle_best(fm)
        if hip is not None:
            rom_raw = rom_percent(hip, self.cfg["top_deg"], self.cfg["bottom_deg"])
            self._rom_smooth = ema(self._rom_smooth, rom_raw, self.cfg["ema_alpha_rom"]) if rom_raw is not None else self._rom_smooth
        else:
            rom_raw = None

        self._vel = velocity(self._rom_smooth, self._last_rom_smooth, dt) if self._rom_smooth is not None else None
        self._last_rom_smooth = self._rom_smooth if self._rom_smooth is not None else self._last_rom_smooth

        self._accumulate(fm, self._rom_smooth if self._rom_smooth is not None else rom_raw)

        prev_stage = self.stage
        self.stage, self.db = stage_update(
            prev_stage=prev_stage, rom=self._rom_smooth, vel=self._vel,
            now_s=now_s, cfg=self.cfg, db=self.db
        )

        # tempo
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
                "exercise": "situp",
                "attempt_index": self.attempt_count,
                "rep_index_valid": self.rep_count if counted else None,
                "counted": counted,
                "class": rep_class,
                "cues": cues,
                "snapshot": {
                    "rom_peak_pct": self._rom_max,
                    "rom_top_pct": self._rom_min,
                    "torso_tilt_peak_deg": self._tilt_peak,
                    "tempo_down_s": round(self._time_down_s, 3),
                    "tempo_up_s": round(self._time_up_s, 3),
                },
            }
            self._reset_attempt()

        live = {
            "stage": self.stage,
            "rep_count": self.rep_count,
            "attempt_count": self.attempt_count,
            "rom": self._rom_smooth if self._rom_smooth is not None else rom_raw,
            "vel": self._vel,
        }
        return attempt_event, live

__all__ = ["SitupDetector", "CFG"]
