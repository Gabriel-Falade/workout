import math
import numpy as np
from typing import Iterable, Optional, Tuple

def angle_2d(a: Iterable[float], b: Iterable[float], c: Iterable[float]) -> Optional[float]:
    """
    Compute the planar angle (in degrees) at point b formed by a–b–c, in the image plane.
    Returns a value in [0, 180]. Returns None if inputs are invalid or degenerate.

    Parameters
    ----------
    a, b, c : (x, y)[, ...]
        Coordinates in normalized image space (0..1). Extra components (e.g., z, visibility)
        are ignored; only the first two values are used.

    Returns
    -------
    float | None
        Angle at b in degrees within [0, 180], or None if any input is invalid.

    Examples
    --------
    >>> angle_2d((0,0), (1,0), (2,0))  # straight line
    180.0
    >>> round(angle_2d((0,0), (0,0), (0,1)) or -1, 1)  # degenerate (ba length = 0)
    -1
    >>> angle_2d((0,0), (1,0), (1,1))  # right angle
    90.0
    """
    # Extract x,y and convert to float arrays
    try:
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        cx, cy = float(c[0]), float(c[1])
    except Exception:
        return None

    # Check for NaNs
    if any(map(math.isnan, (ax, ay, bx, by, cx, cy))):
        return None

    # Vectors from b to a and b to c
    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by

    # Guard against zero-length vectors (undefined angle)
    norm_ba = math.hypot(bax, bay)
    norm_bc = math.hypot(bcx, bcy)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return None

    # Angle via atan2 difference (robust to quadrant issues)
    theta = math.degrees(math.atan2(bcy, bcx) - math.atan2(bay, bax))
    angle = abs(theta) % 360.0
    if angle > 180.0:
        angle = 360.0 - angle

    return angle



def distance_2d(p: Iterable[float], q: Iterable[float]) -> float:
    """
    Compute Euclidean distance between two 2D points in normalized image coordinates.
    Returns None if inputs are invalid.

    Parameters
    ----------
    p, q : (x, y)[, ...]
        Points with at least two values. Extra components (z, visibility) are ignored.

    Returns
    -------
    float | None
        Distance in normalized units (0..√2). None if invalid.
    """
    try:
        px, py = float(p[0]), float(p[1])
        qx, qy = float(q[0]), float(q[1])
    except Exception:
        return None
    
    if any(map(math.isnan, (px, py, qx, qy))):
        return None
    
    dx = qx - px #Calculations for distance x
    dy = qy - py #Calculations for distance y 
    return math.hypot(dx, dy)

def ema(prev: Optional[float], x: Optional[float], alpha: float = 0.2) -> Optional[float]:
    """
    Exponential Moving Average (EMA) for smoothing noisy signals (angles, ROM, positions).

    Parameters
    ----------
    prev : float | None
        Previous EMA value. Use None for the very first sample.
    x : float | None
        Current raw value (e.g., current elbow angle). If None/NaN, EMA holds previous value.
    alpha : float
        Smoothing factor in (0, 1]. Higher = more responsive, lower = smoother.

    Returns
    -------
    float | None
        New EMA value, or None if both prev and x are None/invalid.
    """
    # Clamp alpha to sensible range
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        alpha = 1e-6
    elif alpha > 1:
        alpha = 1.0

    # If x is missing or NaN, just keep the previous EMA
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return prev

    # First valid sample seeds the EMA
    if prev is None:
        return float(x)

    # Normal EMA update
    return float(alpha * x + (1.0 - alpha) * prev)
    
# ROM -> range of motion 
def rom_percent(angle: float, top_angle: float, bottom_angle: float) -> Optional[float]:
    """
    Normalize a joint angle into a 0–100% range of motion where:
      - 0% = TOP / lockout (angle at the top position)
      - 100% = BOTTOM / deepest (angle at the bottom position)

    This works regardless of whether top_angle > bottom_angle or not.
    Values are clamped to [0, 100].

    Parameters
    ----------
    angle : float
        Current (optionally smoothed) joint angle in degrees.
    top_angle : float
        Angle measured at the TOP / lockout position.
    bottom_angle : float
        Angle measured at the BOTTOM / deepest position.

    Returns
    -------
    float | None
        ROM% in [0.0, 100.0], or None if inputs invalid.
    """
    if angle is None:
        return None
    if not isinstance(angle, (int, float)) or not isinstance(top_angle, (int, float)) or not isinstance(bottom_angle, (int, float)):
        return None
    if any(map(math.isnan, (float(angle), float(top_angle), float(bottom_angle)))):
        return None
    if top_angle == bottom_angle:
        return None

    # Clamp angle within the span of the two reference angles
    lo = min(top_angle, bottom_angle)
    hi = max(top_angle, bottom_angle)
    a = min(max(angle, lo), hi)

    # 0 at top, 100 at bottom (sign-safe even when top<bottom)
    rom = 100.0 * (top_angle - a) / (top_angle - bottom_angle)

    # Numerical safety clamp
    if rom < 0.0:
        rom = 0.0
    elif rom > 100.0:
        rom = 100.0

    return rom


def velocity(curr: float, prev: float, dt: float = 1.0) -> Optional[float]:
    """
    Compute velocity (rate of change) between two values.

    Parameters
    ----------
    curr : float
        Current value (angle, ROM%, etc).
    prev : float
        Previous value.
    dt : float
        Time delta (default = 1.0 frame units).

    Returns
    -------
    float | None
        Rate of change, or None if invalid inputs.
    """
    if curr is None or prev is None:
        return None

    if isinstance(curr, float) and math.isnan(curr):
        return None
    if isinstance(prev, float) and math.isnan(prev):
        return None

    if dt <= 0:
        return None

    return (curr - prev) / dt

def stage_update(
    prev_stage: Optional[str],          # "UP" | "DOWN" | None
    rom: Optional[float],               # 0..100 (smoothed)
    vel: Optional[float],               # + = going up (toward 0%), - = going down (toward 100%)
    now_s: float,                       # current timestamp (seconds)
    cfg: dict,                          # thresholds/timings
    db: Optional[dict]                  # debounce state
) -> Tuple[Optional[str], dict]:
    """
    Hysteresis + debounce stage detector.
    Conventions: 0% ROM = top/lockout ("UP"), 100% ROM = bottom/deepest ("DOWN").
    """
    # --- defaults ---
    down_enter = cfg.get("down_enter", 70.0)  # enter DOWN if rom >= this
    down_exit  = cfg.get("down_exit",  55.0)  # stay in DOWN while rom >= this
    up_enter   = cfg.get("up_enter",   10.0)  # enter UP if rom <= this
    up_exit    = cfg.get("up_exit",    20.0)  # stay in UP while rom <= this
    min_speed  = cfg.get("min_speed",   0.0)  # require |vel| >= this to allow switching; 0 disables
    hold_ms    = cfg.get("hold_ms",    150)   # condition must hold this many ms

    # init debounce state
    if db is None:
        db = {"candidate": None, "since_s": None, "last_stage": prev_stage}
    else:
        # ensure required keys exist
        db.setdefault("candidate", None)
        db.setdefault("since_s", None)
        db.setdefault("last_stage", prev_stage)

    # Guard invalid rom
    if rom is None:
        return prev_stage, db

    # Helper: speed gate
    speed_ok = True
    if min_speed > 0.0:
        speed_ok = (vel is not None) and (abs(vel) >= float(min_speed))

    # Determine which stage (if any) is ELIGIBLE now, based on hysteresis + speed
    eligible = None  # "UP" or "DOWN" or None

    if prev_stage in (None, "UP"):
        # Can enter DOWN when deep enough (and speed ok)
        if rom >= down_enter and speed_ok:
            eligible = "DOWN"
        # Forced stay in UP while rom <= up_exit
        elif rom <= up_exit:
            eligible = "UP"  # explicitly reinforce UP region to reset candidate if needed

    if prev_stage == "DOWN":
        # Can enter UP when high enough (and speed ok)
        if rom <= up_enter and speed_ok:
            eligible = "UP"
        # Forced stay in DOWN while rom >= down_exit
        elif rom >= down_exit:
            eligible = "DOWN"

    # Debounce logic
    committed_stage = prev_stage
    if eligible is None or eligible == prev_stage:
        # Either not eligible to switch, or we're already in that stage → clear candidate
        db["candidate"] = None
        db["since_s"] = None
    else:
        # Considering a switch to `eligible`
        if db["candidate"] != eligible:
            # New candidate stage; start timing
            db["candidate"] = eligible
            db["since_s"] = now_s
        else:
            # Same candidate; check hold time
            hold_s = max(0.0, float(hold_ms) / 1000.0)
            if db["since_s"] is not None and (now_s - db["since_s"]) >= hold_s:
                committed_stage = eligible
                # reset debounce state after committing
                db["candidate"] = None
                db["since_s"] = None

    db["last_stage"] = committed_stage
    return committed_stage, db

def midpoint_2d(p: tuple, q: tuple) -> Optional[Tuple[float, float]]:
    """
    Compute the 2D midpoint between two points.

    Parameters
    ----------
    p, q : (x, y)[, ...]
        Points with at least 2 values. Extra components are ignored.

    Returns
    -------
    (x_mid, y_mid) or None if invalid
    """
    try:
        px, py = float(p[0]), float(p[1])
        qx, qy = float(q[0]), float(q[1])
    except Exception:
        return None

    if any(map(math.isnan, (px, py, qx, qy))):
        return None

    return ((px + qx) / 2.0, (py + qy) / 2.0)

def _get_xy(lm: dict, name: str) -> Optional[Tuple[float, float]]:
    """
    Safe getter for a landmark's (x, y).
    Returns None if missing or invalid.
    """
    p = lm.get(name)
    if p is None:
        return None
    try:
        x, y = float(p[0]), float(p[1])
    except Exception:
        return None
    if math.isnan(x) or math.isnan(y):
        return None
    return (x, y)


def shoulder_mid(lm: dict) -> Optional[Tuple[float, float]]:
    """
    Midpoint between L_SHOULDER and R_SHOULDER (x, y).
    """
    ls = _get_xy(lm, "L_SHOULDER")
    rs = _get_xy(lm, "R_SHOULDER")
    return midpoint_2d(ls, rs)


def hip_mid(lm: dict) -> Optional[Tuple[float, float]]:
    """
    Midpoint between L_HIP and R_HIP (x, y).
    """
    lh = _get_xy(lm, "L_HIP")
    rh = _get_xy(lm, "R_HIP")
    return midpoint_2d(lh, rh)


def wrist_rel_shoulder_y(side: str, lm: dict, normalize_by: str = "torso") -> Optional[float]:
    """
    Vertical wrist position relative to shoulder.
    Positive when wrist is ABOVE the shoulder (remember: image y grows downward).

    normalize_by:
      - "torso": divide by shoulder_mid↔hip_mid distance (recommended; robust)
      - "none":  raw delta in normalized image units

    Returns:
      float (unitless) or None if invalid.
      Example: +0.20 means wrist is ~0.20 torso-lengths above shoulder_mid.
    """
    side = side.lower()
    if side not in ("left", "right"):
        return None

    wrist_name = "R_WRIST" if side == "right" else "L_WRIST"
    shoulder_name = "R_SHOULDER" if side == "right" else "L_SHOULDER"

    w = _get_xy(lm, wrist_name)
    s = _get_xy(lm, shoulder_name)
    if w is None or s is None:
        return None

    # delta_y: shoulder_y - wrist_y (positive if wrist is above shoulder)
    dy = s[1] - w[1]

    if normalize_by == "none":
        return dy

    # Default: normalize by torso length (shoulder_mid ↔ hip_mid)
    sm = shoulder_mid(lm)
    hm = hip_mid(lm)
    if sm is None or hm is None:
        return None

    torso_len = distance_2d(sm, hm)
    if torso_len is None or torso_len < 1e-6:
        return None

    return dy / torso_len


def midfoot_x(lm: dict) -> Optional[float]:
    """
    X-coordinate of the midpoint between left/right FOOT_INDEX.
    Useful as a balance / bar-over-midfoot proxy.
    """
    lf = _get_xy(lm, "L_FOOT_INDEX")
    rf = _get_xy(lm, "R_FOOT_INDEX")
    m = midpoint_2d(lf, rf)
    if m is None:
        return None
    return m[0]
