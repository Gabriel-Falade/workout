# src/exercises/burpee.py
from typing import Optional, Dict, List, Tuple
import time
from analysis.metrics import stage_update

CFG = {
    # Squat-down depth
    "squat_depth_target_pct": 70.0,
    # Push-up depth (optional)
    "require_pushup": True,
    "push_depth_target_pct": 85.0,
    # Lockouts
    "stand_lockout_pct": 15.0,
    "push_lockout_pct": 15.0,
    # Flight detection
    "hip_up_vel_thresh":   -120.0,
    "hip_down_vel_thresh":  120.0,
    "flight_window_ms":     300,
}

def _hip_y(lm: Dict) -> Optional[float]:
    lh = lm.get("L_HIP"); rh = lm.get("R_HIP")
    if not lh or not rh: return None
    return 0.5*(float(lh[1]) + float(rh[1]))

class BurpeeDetector:
    """
    Stages: STAND (UP) -> SQUAT (DOWN) -> PUSH (DOWN) -> STAND/JUMP (UP)
    Uses rom_squat_smooth, rom_pushup_smooth; detects jump at end via hip velocity.
    Counts only if depth(s) reached and jump detected (and lockouts).
    """
    def __init__(self, cfg: Dict=None):
        self.cfg = {**CFG, **(cfg or {})}
        self.stage: Optional[str] = "UP"
        self.db=None
        self.rep_count=0; self.attempt_count=0

        self._squat_peak=None
        self._push_peak=None
        self._top_squat=None
        self._top_push=None

        self._prev_hip_y=None
        self._prev_ts=None
        self._arm_ts=None  # flight arm ts

    def _flight_update(self, lm: Dict, now_s: float) -> bool:
        hy = _hip_y(lm)
        if hy is None:
            return False
        if self._prev_ts is None:
            self._prev_ts=now_s; self._prev_hip_y=hy; return False
        dt = max(1e-3, now_s - self._prev_ts)
        vy = (hy - self._prev_hip_y)/dt
        self._prev_hip_y=hy; self._prev_ts=now_s
        if vy <= self.cfg["hip_up_vel_thresh"]:
            self._arm_ts = now_s
        elif vy >= self.cfg["hip_down_vel_thresh"] and self._arm_ts:
            if (now_s - self._arm_ts)*1000.0 <= self.cfg["flight_window_ms"]:
                self._arm_ts=None
                return True
        if self._arm_ts and (now_s - self._arm_ts)*1000.0 > self.cfg["flight_window_ms"]:
            self._arm_ts=None
        return False

    def update(self, fm: Dict, lm: Dict, now_s: Optional[float]=None):
        if now_s is None: now_s = time.time()
        rs = fm.get("rom_squat_smooth"); rp = fm.get("rom_pushup_smooth")
        vs = fm.get("vel_squat"); vp = fm.get("vel_pushup")

        # track extrema
        if rs is not None:
            self._squat_peak = rs if self._squat_peak is None else max(self._squat_peak, rs)
            self._top_squat  = rs if self._top_squat  is None else min(self._top_squat,  rs)
        if rp is not None:
            self._push_peak = rp if self._push_peak is None else max(self._push_peak, rp)
            self._top_push  = rp if self._top_push  is None else min(self._top_push,  rp)

        # very simple state: rely on DOWN/UP from squat ROM
        prev = self.stage
        stage, _ = stage_update("DOWN" if prev=="DOWN" else "UP", rs, vs, now_s,
                                {"down_enter":60.0,"down_exit":45.0,"up_enter":10.0,"up_exit":20.0,
                                 "hold_ms":120,"min_speed":6.0}, None)
        self.stage = "DOWN" if stage=="DOWN" else "UP"

        event=None
        flight = self._flight_update(lm, now_s)

        # count at UP after a full cycle with required features
        if prev == "DOWN" and self.stage == "UP":
            self.attempt_count += 1
            cls="good"; counted=True; cues=[]

            if (self._squat_peak or -1) < self.cfg["squat_depth_target_pct"]:
                cls, counted = "squat_depth_short", False; cues.append("Squat deeper")

            if self.cfg["require_pushup"]:
                if (self._push_peak or -1) < self.cfg["push_depth_target_pct"]:
                    cls, counted = "pushup_depth_short", False; cues.append("Hit push-up depth")
                if (self._top_push or 1e9) > self.cfg["push_lockout_pct"]:
                    if cls=="good": cls, counted = "pushup_lockout_fail", False; cues.append("Finish push-up lockout")

            if (self._top_squat or 1e9) > self.cfg["stand_lockout_pct"]:
                if cls=="good": cls, counted = "stand_lockout_fail", False; cues.append("Stand tall before jump")

            if cls=="good" and not flight:
                cls, counted = "no_jump_detected", False; cues.append("Explode upward—jump at finish")

            if counted: self.rep_count += 1
            event={"ts": now_s, "exercise":"burpee",
                   "attempt_index": self.attempt_count,
                   "rep_index_valid": self.rep_count if counted else None,
                   "counted": counted, "class": cls, "cues": cues[:2],
                   "snapshot":{"squat_peak": self._squat_peak, "push_peak": self._push_peak,
                               "top_squat": self._top_squat, "top_push": self._top_push,
                               "flight": bool(flight)}}
            self._squat_peak=self._push_peak=self._top_squat=self._top_push=None

        live={"stage": self.stage, "rep_count": self.rep_count, "attempt_count": self.attempt_count,
              "rom": rs, "vel": vs}
        return event, live

__all__ = ["BurpeeDetector", "CFG"]
