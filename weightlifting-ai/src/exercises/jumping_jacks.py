# src/exercises/jumping_jacks.py
from typing import Optional, Dict, List, Tuple
import time
from analysis.metrics import stage_update

CFG = {
    # Use simple open/closed gates from wrist height and foot width
    "arms_up_thresh":    0.12,   # wrist_rel_shoulder_y ≥ this (torso-normalized) counts as "up"
    "arms_down_thresh": -0.02,   # ≤ this counts as "down"
    "feet_apart_thresh": 0.12,   # |L_ANKLE.x - R_ANKLE.x| ≥ this counts as "apart"
    "feet_together_thr": 0.06,   # ≤ this counts as "together"

    # Stage machine on a synthesized ROM (0 closed, 100 open)
    "down_enter": 35.0, "down_exit": 25.0,
    "up_enter":   65.0, "up_exit":   55.0,
    "hold_ms":   120, "min_speed": 12.0,
}

def _feet_width(lm: Dict) -> Optional[float]:
    la = lm.get("L_ANKLE"); ra = lm.get("R_ANKLE")
    if not la or not ra: return None
    try:
        return abs(float(la[0]) - float(ra[0]))
    except Exception:
        return None

class JumpingJacksDetector:
    """
    Uses combined arms/feet signal. ROM proxy:
      0% = arms down & feet together (closed)
      100% = arms up & feet apart (open)
    Counts on OPEN -> CLOSED (or CLOSED -> OPEN; choose below).
    """
    def __init__(self, cfg: Dict=None):
        self.cfg = {**CFG, **(cfg or {})}
        self.stage: Optional[str] = "DOWN"  # start closed
        self.db: Optional[Dict] = None
        self.rep_count = 0
        self.attempt_count = 0
        self._rom = 0.0
        self._last_ts: Optional[float] = None

    def _compute_rom(self, fm: Dict, lm: Dict) -> Optional[float]:
        # Arms: take max wrist height (best side)
        wr = fm.get("wrist_rel_y_r"); wl = fm.get("wrist_rel_y_l")
        arms_score = max([v for v in (wr, wl) if v is not None], default=None)
        feet = _feet_width(lm)
        if arms_score is None or feet is None:
            return None

        # Normalize both to [0,1]-ish, then blend
        arms_open = 1.0 if arms_score >= self.cfg["arms_up_thresh"] else \
                    0.0 if arms_score <= self.cfg["arms_down_thresh"] else \
                    (arms_score - self.cfg["arms_down_thresh"]) / (self.cfg["arms_up_thresh"] - self.cfg["arms_down_thresh"])

        feet_open = 1.0 if feet >= self.cfg["feet_apart_thresh"] else \
                    0.0 if feet <= self.cfg["feet_together_thr"] else \
                    (feet - self.cfg["feet_together_thr"]) / (self.cfg["feet_apart_thresh"] - self.cfg["feet_together_thr"])

        rom = 100.0 * (0.6 * arms_open + 0.4 * feet_open)  # weight arms more
        return max(0.0, min(100.0, rom))

    def update(self, fm: Dict, lm: Dict, now_s: Optional[float]=None):
        if now_s is None: now_s = time.time()
        rom = self._compute_rom(fm, lm)
        vel = None  # not used here; we rely on stage hysteresis

        prev = self.stage
        self.stage, self.db = stage_update(prev, rom, vel, now_s, self.cfg, self.db)

        event = None
        if prev == "UP" and self.stage == "DOWN":   # count on close
            self.attempt_count += 1
            self.rep_count += 1
            event = {"ts": now_s, "exercise": "jumping_jacks",
                     "attempt_index": self.attempt_count, "rep_index_valid": self.rep_count,
                     "counted": True, "class": "good", "cues": [], "snapshot": {"rom": rom}}

        live = {"stage": self.stage, "rep_count": self.rep_count, "attempt_count": self.attempt_count,
                "rom": rom, "vel": None}
        return event, live

__all__ = ["JumpingJacksDetector", "CFG"]
