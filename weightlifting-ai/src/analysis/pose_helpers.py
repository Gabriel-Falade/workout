import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from .metrics import angle_2d, midpoint_2d, distance_2d
import mediapipe as mp

def visibility_ok(landmarks: Dict[str, Tuple[float, float, float, float]],
                  names: List[str],
                  thresh: float = 0.5) -> bool:
    """
    Check that all required landmarks exist and have visibility >= threshold.

    Parameters
    ----------
    landmarks : dict
        Mapping from landmark name -> (x, y, z?, visibility).
    names : list of str
        Landmark names to check.
    thresh : float
        Minimum visibility required (default = 0.5).

    Returns
    -------
    bool
        True if all requested landmarks are present and visible.
    """
    for name in names:
        lm = landmarks.get(name)
        if lm is None:
            return False

        vis = lm[3]
        if vis is None or (isinstance(vis, float) and math.isnan(vis)):
            return False
        if vis < thresh:
            return False

    return True

def elbow_angle(side: str, lm: dict) -> Optional[float]:
    """
    Compute elbow flexion/extension angle for left or right side.

    Parameters
    ----------
    side : str
        "left" or "right"
    lm : dict
        Mapping from landmark name to (x, y, z?, visibility).

    Returns
    -------
    float | None
        Elbow angle in degrees [0, 180], or None if missing/invalid.
    """
    # 1. Get shoulder, elbow, wrist landmarks (depending on side)
    # 2. Call angle_2d(shoulder, elbow, wrist)
    # 3. Return result
    side = side.lower()
    if side not in ("left", "right"):
        return None

    if side == "right":
        return angle_2d(lm.get("R_SHOULDER"), lm.get("R_ELBOW"), lm.get("R_WRIST"))
    else:
        return angle_2d(lm.get("L_SHOULDER"), lm.get("L_ELBOW"), lm.get("L_WRIST"))
    
def knee_angle(side: str, lm: dict) -> Optional[float]:
    """
    Compute knee flexion/extension angle for left or right side.

    Parameters
    ----------
    side : str
        "left" or "right"
    lm : dict
        Mapping from landmark name to (x, y, z?, visibility).

    Returns
    -------
    float | None
        Knee angle in degrees [0, 180], or None if missing/invalid.
    """
    side = side.lower()
    if side not in ("left", "right"):
        return None

    if side == "right":
        return angle_2d(lm.get("R_HIP"), lm.get("R_KNEE"), lm.get("R_ANKLE"))
    else:
        return angle_2d(lm.get("L_HIP"), lm.get("L_KNEE"), lm.get("L_ANKLE"))
    
def hip_angle(side: str, lm: dict) -> Optional[float]:
    """
    Compute hip flexion/extension angle for left or right side.

    Parameters
    ----------
    side : str
        "left" or "right"
    lm : dict
        Mapping from landmark name to (x, y, z?, visibility).

    Returns
    -------
    float | None
        Hip angle in degrees [0, 180], or None if missing/invalid.
    """
    side = side.lower()
    if side not in ("left", "right"):
        return None

    if side == "right":
        return angle_2d(lm.get("R_SHOULDER"), lm.get("R_HIP"), lm.get("R_KNEE"))
    else:
        return angle_2d(lm.get("L_SHOULDER"), lm.get("L_HIP"), lm.get("L_KNEE"))
    

def shoulder_angle(side: str, lm: dict) -> Optional[float]:
    """
    Shoulder abduction/flexion proxy: angle at the SHOULDER formed by (HIP SHOULDER ELBOW).
    Degrees in [0, 180]. Returns None if missing/invalid.

    side: "left" | "right"
    lm:   dict with keys "L_HIP","L_SHOULDER","L_ELBOW","R_HIP","R_SHOULDER","R_ELBOW"
    """
    side = side.lower()
    if side not in ("left", "right"):
        return None
    if side == "right":
        return angle_2d(lm.get("R_HIP"), lm.get("R_SHOULDER"), lm.get("R_ELBOW"))
    else:
        return angle_2d(lm.get("L_HIP"), lm.get("L_SHOULDER"), lm.get("L_ELBOW"))


def ankle_angle(side: str, lm: dict) -> Optional[float]:
    """
    Ankle plantar/dorsi-flexion proxy: angle at the ANKLE formed by (KNEE ANKLE FOOT_INDEX).
    Degrees in [0, 180]. Returns None if missing/invalid.

    side: "left" | "right"
    lm:   dict with keys "L_KNEE","L_ANKLE","L_FOOT_INDEX","R_KNEE","R_ANKLE","R_FOOT_INDEX"
    """
    side = side.lower()
    if side not in ("left", "right"):
        return None
    if side == "right":
        return angle_2d(lm.get("R_KNEE"), lm.get("R_ANKLE"), lm.get("R_FOOT_INDEX"))
    else:
        return angle_2d(lm.get("L_KNEE"), lm.get("L_ANKLE"), lm.get("L_FOOT_INDEX"))
    

def torso_tilt(lm: dict) -> Optional[float]:
    """
    Compute torso tilt (lean angle) relative to vertical, using midpoints of shoulders and hips.

    Returns
    -------
    float | None
        Tilt angle in degrees [0, 90], or None if invalid.
    """
    r_shoulder, l_shoulder = lm.get("R_SHOULDER"), lm.get("L_SHOULDER")
    r_hip, l_hip = lm.get("R_HIP"), lm.get("L_HIP")

    shoulder_mid = midpoint_2d(r_shoulder, l_shoulder)
    hip_mid = midpoint_2d(r_hip, l_hip)
    if shoulder_mid is None or hip_mid is None:
        return None

    # Vector from hip to shoulder
    vec = (shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
    mag_vec = math.hypot(*vec)
    if mag_vec < 1e-8:
        return None

    # Vertical reference vector
    vertical = (0.0, -1.0)  # careful: in image coords, y increases downward
    mag_vert = math.hypot(*vertical)

    # Cosine similarity
    cos_theta = (vec[0]*vertical[0] + vec[1]*vertical[1]) / (mag_vec * mag_vert)
    cos_theta = max(-1.0, min(1.0, cos_theta))  # clamp

    # Convert to degrees
    angle = math.degrees(math.acos(cos_theta))
    return min(angle, 90.0)

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

def knee_valgus(side: str, lm: dict, normalize: bool = True) -> Optional[float]:
    """
    2D proxy for knee valgus: knee's medial/lateral offset from the hip–ankle line.
    Negative → knee inside (valgus), Positive → knee outside (varus).

    If normalize=True, returns offset divided by hip–ankle segment length (unitless).
    """
    side = side.lower()
    if side not in ("left", "right"):
        return None

    knee_name  = "R_KNEE"  if side == "right" else "L_KNEE"
    hip_name   = "R_HIP"   if side == "right" else "L_HIP"
    ankle_name = "R_ANKLE" if side == "right" else "L_ANKLE"

    k = _get_xy(lm, knee_name); h = _get_xy(lm, hip_name); a = _get_xy(lm, ankle_name)
    if k is None or h is None or a is None:
        return None

    hx, hy = h; ax, ay = a; kx, ky = k
    vx, vy = ax - hx, ay - hy
    vnorm2 = vx*vx + vy*vy
    if vnorm2 < 1e-6:
        return None

    # projection of K onto HA
    t = ((kx - hx)*vx + (ky - hy)*vy) / vnorm2
    px, py = hx + t*vx, hy + t*vy

    offset = kx - px  # medial/lateral in image x
    if not normalize:
        return offset

    seg_len = math.sqrt(vnorm2)
    if seg_len < 1e-6:
        return None
    return offset / seg_len


def body_height_proxy(lm: dict) -> Optional[float]:
    """
    Approximate body height in normalized image units.
    Uses shoulder_mid ↔ ankle_mid distance.

    Returns:
        Float (0..1) or None if invalid.
    """
    sm = shoulder_mid(lm)
    lh = _get_xy(lm, "L_ANKLE")
    rh = _get_xy(lm, "R_ANKLE")
    am = midpoint_2d(lh, rh)

    if sm is None or am is None:
        return None

    return distance_2d(sm, am)

def torso_tilt_signed(lm: dict) -> Optional[float]:
    """
    Signed torso tilt relative to vertical using shoulder_mid → hip_mid vector.
    Positive = leaning FORWARD (shoulders ahead), Negative = leaning BACKWARD.

    Returns degrees in [-90, +90]. None if invalid.
    """
    r_sh, l_sh = lm.get("R_SHOULDER"), lm.get("L_SHOULDER")
    r_hip, l_hip = lm.get("R_HIP"), lm.get("L_HIP")
    sm = midpoint_2d(r_sh, l_sh); hm = midpoint_2d(r_hip, l_hip)
    if sm is None or hm is None:
        return None

    # vector hip→shoulder (x right+, y down+)
    vx, vy = sm[0] - hm[0], sm[1] - hm[1]
    mag = math.hypot(vx, vy)
    if mag < 1e-8:
        return None

    # angle from vertical: vertical vector is (0, -1)
    cos_theta = (vx*0.0 + vy*(-1.0)) / mag
    cos_theta = max(-1.0, min(1.0, cos_theta))
    unsigned = math.degrees(math.acos(cos_theta))  # 0..180

    # sign by horizontal component: shoulders ahead (vx>0 in image coords after mirroring?)
    # NOTE: sign relies on camera/mirroring; use with care.
    sign = 1.0 if vx > 0 else -1.0 if vx < 0 else 0.0
    angle = unsigned if sign >= 0 else -unsigned
    # clamp to [-90, 90]
    return max(-90.0, min(90.0, angle))

_L = mp.solutions.pose.PoseLandmark

def mp_results_to_dict(results) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Convert Mediapipe Pose results into: name -> (x, y, z, visibility)
    Returns {} if results/landmarks are missing.
    """
    lm_dict: Dict[str, Tuple[float, float, float, float]] = {}
    if not results or not results.pose_landmarks:
        return lm_dict

    lm = results.pose_landmarks.landmark
    def g(idx): 
        p = lm[idx]
        return (float(p.x), float(p.y), float(getattr(p, "z", 0.0)), float(getattr(p, "visibility", 0.0)))

    names = {
        "NOSE": _L.NOSE, "L_SHOULDER": _L.LEFT_SHOULDER, "R_SHOULDER": _L.RIGHT_SHOULDER,
        "L_ELBOW": _L.LEFT_ELBOW, "R_ELBOW": _L.RIGHT_ELBOW, "L_WRIST": _L.LEFT_WRIST, "R_WRIST": _L.RIGHT_WRIST,
        "L_HIP": _L.LEFT_HIP, "R_HIP": _L.RIGHT_HIP, "L_KNEE": _L.LEFT_KNEE, "R_KNEE": _L.RIGHT_KNEE,
        "L_ANKLE": _L.LEFT_ANKLE, "R_ANKLE": _L.RIGHT_ANKLE, "L_FOOT_INDEX": _L.LEFT_FOOT_INDEX, "R_FOOT_INDEX": _L.RIGHT_FOOT_INDEX,
    }
    for k, v in names.items():
        lm_dict[k] = g(v.value if hasattr(v, "value") else v)
    return lm_dict
