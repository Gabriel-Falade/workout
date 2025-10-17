# src/exercises/high_knees.py
from typing import Optional, Dict, List, Tuple
import time

CFG = {
    "knee_above_hip_thresh": 0.10,  # knee.y above hip.y by ≥ this *torso-lengths*
    "debounce_ms": 220,
}

class HighKneesCounter:
    """
    Counts alternating knee raises. No stage machine; per-side edge detectors.
    Needs: hip_mid, knee y's, torso_len. Uses normalized vertical delta.
    """
    def __init__(self, cfg: Dict=None):
        self.cfg = {**CFG, **(cfg or {})}
        self.rep_count = 0
        self._last_hit_side: Optional[str] = None
        self._last_hit_ts: float = 0.0

    def _normalized_dy(self, fm: Dict, lm: Dict, side: str) -> Optional[float]:
        knee = lm.get("R_KNEE") if side=="right" else lm.get("L_KNEE")
        hipm = (fm.get("hip_mid_x"), fm.get("hip_mid_y"))
        torso_len = fm.get("torso_len")
        if not knee or hipm[0] is None or hipm[1] is None or not torso_len or torso_len < 1e-6:
            return None
        # Positive when knee is ABOVE the hip
        return (hipm[1] - float(knee[1])) / torso_len

    def update(self, fm: Dict, lm: Dict, now_s: Optional[float]=None):
        if now_s is None: now_s = time.time()

        hits = []
        for side in ("left", "right"):
            dy = self._normalized_dy(fm, lm, side)
            if dy is not None and dy >= self.cfg["knee_above_hip_thresh"]:
                hits.append(side)

        event = None
        if hits:
            # prefer the opposite side of last hit to avoid double-counting same side
            side = hits[0]
            if self._last_hit_side == side and len(hits) == 2:
                side = "left" if side == "right" else "right"

            if (now_s - self._last_hit_ts)*1000.0 >= self.cfg["debounce_ms"]:
                if self._last_hit_side != side:
                    self.rep_count += 1
                    event = {"ts": now_s, "exercise": "high_knees",
                             "rep_index_valid": self.rep_count, "counted": True,
                             "class": "good", "cues": [], "snapshot": {"side": side}}
                    self._last_hit_side = side
                    self._last_hit_ts = now_s

        live = {"rep_count": self.rep_count, "stage": None, "attempt_count": self.rep_count, "rom": None, "vel": None}
        return event, live

__all__ = ["HighKneesCounter", "CFG"]
