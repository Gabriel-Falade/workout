# src/exercises/jump_squat.py
from typing import Optional, Dict, List, Tuple
import time
from analysis.metrics import stage_update

CFG = {
    "down_enter": 60.0, "down_exit": 45.0,
    "up_enter":   10.0, "up_exit":   20.0,
    "hold_ms":   120, "min_speed": 8.0,

    "depth_target_pct": 70.0,  # not as deep as squat; it's plyometric
    "lockout_target_pct": 18.0,

    # flight detection
    "hip_up_vel_thresh":   -120.0,  # (remember: image y grows downward; up is negative vel) tuned per FPS
    "hip_down_vel_thresh":  120.0,
    "flight_window_ms":     280,
    "heels_sep_delta":      0.02,   # if heels briefly separate vertically, helps confirm
}

def _hip_y(lm: Dict) -> Optional[float]:
    lh = lm.get("L_HIP"); rh = lm.get("R_HIP")
    if not lh or not rh: return None
    return 0.5*(float(lh[1]) + float(rh[1]))

def _heels_y(lm: Dict) -> Optional[Tuple[float, float]]:
    lh = lm.get("L_HEEL") or lm.get("L_ANKLE")
    rh = lm.get("R_HEEL") or lm.get("R_ANKLE")
    if not lh or not rh: return None
    return float(lh[1]), float(rh[1])

class JumpSquatDetector:
    """
    Requires: rom_squat_smooth, vel_squat, plus raw lm for hip/heel y checks.
    Counts when: depth reached, then on UP near top detect "flight" (hip y up then down within window).
    """
    def __init__(self, cfg: Dict=None):
        self.cfg = {**CFG, **(cfg or {})}
        self.stage: Optional[str] = None
        self.db: Optional[Dict] = None
        self.rep_count=0; self.attempt_count=0

        self._rom_min=None; self._rom_max=None
        self._flight_arm_ts: Optional[float]=None

        self._prev_hip_y: Optional[float]=None
        self._prev_ts: Optional[float]=None

    def _update_flight_detector(self, lm: Dict, now_s: float) -> bool:
        # crude: watch hip y velocity sign flip within a small window
        hy = _hip_y(lm)
        if hy is None:
            return False
        if self._prev_ts is None:
            self._prev_ts = now_s; self._prev_hip_y = hy; return False

        dt = max(1e-3, now_s - self._prev_ts)
        vy = (hy - self._prev_hip_y) / dt  # + = moving down; - = moving up
        self._prev_hip_y = hy; self._prev_ts = now_s

        # arm when strong upward (negative) velocity; confirm if strong downward soon after
        if vy <= self.cfg["hip_up_vel_thresh"]:
            self._flight_arm_ts = now_s
        elif vy >= self.cfg["hip_down_vel_thresh"] and self._flight_arm_ts is not None:
            if (now_s - self._flight_arm_ts)*1000.0 <= self.cfg["flight_window_ms"]:
                # optional heels divergence check
                hh = _heels_y(lm)
                if hh:
                    dy = abs(hh[0] - hh[1])
                    if dy >= self.cfg["heels_sep_delta"]:
                        self._flight_arm_ts = None
                        return True
                # accept even without heel cue
                self._flight_arm_ts = None
                return True

        # timeout
        if self._flight_arm_ts and (now_s - self._flight_arm_ts)*1000.0 > self.cfg["flight_window_ms"]:
            self._flight_arm_ts = None
        return False

    def update(self, fm: Dict, lm: Dict, now_s: Optional[float]=None):
        if now_s is None: now_s = time.time()

        rom = fm.get("rom_squat_smooth")
        vel = fm.get("vel_squat")

        # track rom extrema
        if rom is not None:
            self._rom_min = rom if self._rom_min is None else min(self._rom_min, rom)
            self._rom_max = rom if self._rom_max is None else max(self._rom_max, rom)

        prev = self.stage
        self.stage, self.db = stage_update(prev, rom, vel, now_s, self.cfg, self.db)

        flight = self._update_flight_detector(lm, now_s)
        event=None
        if prev == "DOWN" and self.stage == "UP":
            self.attempt_count += 1
            cls="good"; counted=True; cues=[]

            if (self._rom_max or -1) < self.cfg["depth_target_pct"]:
                cls, counted = "depth_short", False; cues.append("Squat deeper before jump")

            if (self._rom_min or 1e9) > self.cfg["lockout_target_pct"]:
                if cls == "good": cls, counted = "lockout_fail", False; cues.append("Finish tall at top")

            if cls=="good" and not flight:
                cls, counted = "no_jump_detected", False; cues.append("Explode upward—aim for airtime")

            if counted: self.rep_count += 1
            event = {"ts": now_s, "exercise": "jump_squat",
                     "attempt_index": self.attempt_count,
                     "rep_index_valid": self.rep_count if counted else None,
                     "counted": counted, "class": cls, "cues": cues[:2],
                     "snapshot": {"rom_peak_pct": self._rom_max, "rom_top_pct": self._rom_min, "flight": bool(flight)}}
            self._rom_min=self._rom_max=None

        live={"stage": self.stage, "rep_count": self.rep_count, "attempt_count": self.attempt_count,
              "rom": rom, "vel": vel}
        return event, live

__all__ = ["JumpSquatDetector", "CFG"]
