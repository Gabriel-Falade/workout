# src/exercises/donkey_kicks.py
from typing import Optional, Dict, List, Tuple
import time
from analysis.metrics import ema, velocity, stage_update
from analysis.pose_helpers import wrist_rel_shoulder_y  # used pattern; here we roll our own ankle rel hip

CFG = {
    "ema_alpha": 0.35,
    "up_enter":   65.0, "up_exit":   50.0,
    "down_enter": 15.0, "down_exit": 25.0,
    "hold_ms":   120, "min_speed": 6.0,
    "height_target": 0.35,   # ankle ≥ this many torso-lengths above hip → good height
}

def _ankle_rel_hip_y(side: str, fm: Dict, lm: Dict) -> Optional[float]:
    """Positive when ankle is ABOVE hip, normalized by torso length."""
    hip = lm.get("R_HIP") if side=="right" else lm.get("L_HIP")
    ank = lm.get("R_ANKLE") if side=="right" else lm.get("L_ANKLE")
    smx, smy = fm.get("shoulder_mid_x"), fm.get("shoulder_mid_y")
    hmx, hmy = fm.get("hip_mid_x"), fm.get("hip_mid_y")
    torso_len = fm.get("torso_len")
    if not hip or not ank or smx is None or hmx is None or torso_len is None or torso_len < 1e-6:
        return None
    return (float(hip[1]) - float(ank[1])) / torso_len

class DonkeyKicksDetector:
    """
    Counts per-side kicks when ankle height (vs hip) reaches a peak and returns.
    Uses a simple ROM proxy: 0 = ankle below hip, 100 = clearly above target.
    """
    def __init__(self, side: str="right", cfg: Dict=None):
        self.side = side.lower()
        self.cfg = {**CFG, **(cfg or {})}
        self.stage: Optional[str] = None
        self.db: Optional[Dict] = None
        self.rep_count = 0
        self.attempt_count = 0
        self._rom_s = None
        self._last_rom_s = None
        self._vel = None
        self._last_ts = None

    def update(self, fm: Dict, lm: Dict, now_s: Optional[float]=None):
        if now_s is None: now_s = time.time()
        if self._last_ts is None: dt = 1/60.0
        else: dt = max(1e-3, now_s - self._last_ts)
        self._last_ts = now_s

        h = _ankle_rel_hip_y(self.side, fm, lm)
        rom = None
        if h is not None:
            # map height to ROM%: 0 at 0.0; 100 at >= height_target
            rom = max(0.0, min(100.0, 100.0 * (h / self.cfg["height_target"])))
            self._rom_s = ema(self._rom_s, rom, self.cfg["ema_alpha"]) if rom is not None else self._rom_s
            self._vel = velocity(self._rom_s, self._last_rom_s, dt) if self._rom_s is not None else None
            self._last_rom_s = self._rom_s if self._rom_s is not None else self._last_rom_s

        prev = self.stage
        self.stage, self.db = stage_update(prev, self._rom_s, self._vel, now_s, self.cfg, self.db)

        event = None
        if prev == "UP" and self.stage == "DOWN":
            self.attempt_count += 1
            self.rep_count += 1
            event = {"ts": now_s, "exercise": f"donkey_kicks_{self.side}",
                     "attempt_index": self.attempt_count, "rep_index_valid": self.rep_count,
                     "counted": True, "class": "good", "cues": [], "snapshot": {"height_norm": h}}

        live = {"stage": self.stage, "rep_count": self.rep_count, "attempt_count": self.attempt_count,
                "rom": self._rom_s if self._rom_s is not None else rom, "vel": self._vel}
        return event, live

__all__ = ["DonkeyKicksDetector", "CFG"]
