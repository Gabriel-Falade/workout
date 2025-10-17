import numpy as np

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



#def distance_2d()

print(angle_2d((0,0), (1,0), (2,0)))