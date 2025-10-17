# src/exercises/plank.py
from typing import Optional, Dict, List, Tuple
import time

CFG = {
    "torso_tilt_max_deg": 20.0,     # keep straight
    "hip_sag_min_deg":   165.0,     # warn if hips < this (sag)
    "elbow_min_deg":      160.0,    # optional: near-straight arm plank
    "emit_every_s":         10.0,   # emit hold event every N seconds of continuous OK
    "grace_ms":            300,     # brief dips ignored
}

class PlankMonitor:
    """
    Isometric plank quality. No reps—tracks continuous 'OK' streak.
    update(fm, now_s) -> (event|None, live)
      fm needs: torso_tilt_deg, r_hip_deg, l_hip_deg, r_elbow_deg, l_elbow_deg
    """
    def __init__(self, cfg: Dict=None):
        self.cfg = {**CFG, **(cfg or {})}
        self._ok_since: Optional[float] = None
        self._last_emit: float = 0.0
        self._violation_since: Optional[float] = None

    def _is_ok(self, fm: Dict) -> Tuple[bool, List[str]]:
        cues = []
        ok = True
        tilt = fm.get("torso_tilt_deg")
        if tilt is None or tilt > self.cfg["torso_tilt_max_deg"]:
            ok = False; cues.append("Reduce torso tilt")

        hips = [fm.get("r_hip_deg"), fm.get("l_hip_deg")]
        hips = [x for x in hips if x is not None]
        if hips and min(hips) < self.cfg["hip_sag_min_deg"]:
            ok = False; cues.append("Avoid hip sag—brace core")

        elbows = [fm.get("r_elbow_deg"), fm.get("l_elbow_deg")]
        elbows = [x for x in elbows if x is not None]
        if elbows and max(elbows) < self.cfg["elbow_min_deg"]:
            # not fatal by default—just warn
            cues.append("Lock elbows a bit more")
        return ok, cues

    def update(self, fm: Dict, now_s: Optional[float]=None):
        if now_s is None:
            now_s = time.time()

        ok, cues = self._is_ok(fm)
        event = None

        if ok:
            # end violation
            self._violation_since = None
            if self._ok_since is None:
                self._ok_since = now_s
            # periodic emit
            held_s = now_s - self._ok_since
            if held_s - self._last_emit >= self.cfg["emit_every_s"]:
                self._last_emit += self.cfg["emit_every_s"]
                event = {
                    "ts": now_s,
                    "exercise": "plank",
                    "class": "hold_tick",
                    "held_s": round(held_s, 1),
                    "cues": cues[:2],
                }
        else:
            # start/extend violation window
            if self._violation_since is None:
                self._violation_since = now_s
            if (now_s - self._violation_since) * 1000.0 >= self.cfg["grace_ms"]:
                # break streak
                if self._ok_since is not None:
                    held_s = now_s - self._ok_since
                else:
                    held_s = 0.0
                event = {
                    "ts": now_s,
                    "exercise": "plank",
                    "class": "hold_break",
                    "held_s": round(held_s, 1),
                    "cues": cues[:2],
                }
                self._ok_since = None
                self._last_emit = 0.0

        live = {
            "hold_ok": ok,
            "held_s": 0.0 if self._ok_since is None else now_s - self._ok_since,
            "violating": not ok,
            "cues": cues[:2],
        }
        return event, live

__all__ = ["PlankMonitor", "CFG"]
