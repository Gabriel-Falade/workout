# src/exercises/glute_bridge.py
from typing import Optional, Dict, List, Tuple
import time
from analysis.metrics import rom_percent, ema, velocity, stage_update

CFG = {
    "top_deg": 175.0,    # hip angle at lockout
    "bottom_deg": 120.0, # hip angle at bottom (hips down)
    "down_enter": 60.0, "down_exit":  45.0,
    "up_enter":   10.0, "up_exit":    20.0,
    "hold_ms":   120, "min_speed": 6.0,

    "quarter_depth_pct":  30.0,
    "half_depth_pct":     55.0,
    "depth_target_pct":   85.0,
    "lockout_target_pct": 12.0,

    "ema_alpha": 0.30,
}

class GluteBridgeDetector:
    """
    Hip-extension ROM via smaller hip angle (more flexed) mapped into 0..100 (down to up).
    Counts when depth and lockout are satisfied.
    """
    def __init__(self, cfg: Dict=None):
        self.cfg = {**CFG, **(cfg or {})}
        self.stage: Optional[str] = None
        self.db: Optional[Dict] = None
        self.rep_count = 0
        self.attempt_count = 0
        self._rom_s = None
        self._last_rom_s = None
        self._vel = None
        self._last_ts = None
        self._rom_min = None; self._rom_max = None

    def _hip_angle_best(self, fm: Dict) -> Optional[float]:
        lh = fm.get("l_hip_deg"); rh = fm.get("r_hip_deg")
        if lh is None and rh is None: return None
        if lh is None: return rh
        if rh is None: return lh
        return min(lh, rh)

    def update(self, fm: Dict, now_s: Optional[float]=None):
        if now_s is None: now_s = time.time()
        if self._last_ts is None: dt = 1/60.0
        else: dt = max(1e-3, now_s - self._last_ts)
        self._last_ts = now_s

        hip = self._hip_angle_best(fm)
        rom_raw = rom_percent(hip, self.cfg["top_deg"], self.cfg["bottom_deg"]) if hip is not None else None
        if rom_raw is not None:
            self._rom_s = ema(self._rom_s, rom_raw, self.cfg["ema_alpha"]) if rom_raw is not None else self._rom_s
            self._vel = velocity(self._rom_s, self._last_rom_s, dt) if self._rom_s is not None else None
            self._last_rom_s = self._rom_s if self._rom_s is not None else self._last_rom_s
            self._rom_min = self._rom_s if self._rom_min is None else min(self._rom_min, self._rom_s)
            self._rom_max = self._rom_s if self._rom_max is None else max(self._rom_max, self._rom_s)

        prev = self.stage
        self.stage, self.db = stage_update(prev, self._rom_s, self._vel, now_s, self.cfg, self.db)

        event=None
        if prev == "DOWN" and self.stage == "UP":
            self.attempt_count += 1
            depth_peak = self._rom_max if self._rom_max is not None else -1.0
            top_rom    = self._rom_min if self._rom_min is not None else 1e9
            cls="good"; counted=True; cues=[]
            if depth_peak < self.cfg["quarter_depth_pct"]:
                cls, counted = "quarter_rep", False; cues.append("Lift hips MUCH higher")
            elif depth_peak < self.cfg["half_depth_pct"]:
                cls, counted = "half_rep", False; cues.append("Lift hips higher")
            elif depth_peak < self.cfg["depth_target_pct"]:
                cls, counted = "depth_short", False; cues.append(f"Hit ≥ {int(self.cfg['depth_target_pct'])}% up")
            lock_ok = top_rom <= self.cfg["lockout_target_pct"]
            if cls=="good" and not lock_ok:
                cls, counted = "lockout_fail", False; cues.append("Return fully to bottom between reps")

            if counted: self.rep_count += 1
            event = {"ts": now_s, "exercise": "glute_bridge",
                     "attempt_index": self.attempt_count,
                     "rep_index_valid": self.rep_count if counted else None,
                     "counted": counted, "class": cls, "cues": cues[:2],
                     "snapshot": {"rom_peak_pct": self._rom_max, "rom_top_pct": self._rom_min}}
            self._rom_min = self._rom_max = None

        live = {"stage": self.stage, "rep_count": self.rep_count, "attempt_count": self.attempt_count,
                "rom": self._rom_s, "vel": self._vel}
        return event, live

__all__ = ["GluteBridgeDetector", "CFG"]
