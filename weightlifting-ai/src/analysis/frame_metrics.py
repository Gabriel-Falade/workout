# frame_metrics.py

"""
One-pass extractor for per-frame pose metrics.

Inputs:
  - lm: dict[str, tuple]  # Mediapipe-style landmarks: (x, y, z?, visibility)
  - dt: float             # seconds since last frame

Outputs (dict):
  - raw angles, mids, tilt, normalized positions
  - derived ROM% per exercise (pushup/squat/ohp)
  - velocities on ROM% (for stage machines)
  - visibility flags and body size proxies

This module is *stateless* by default. To get EMA & velocity across frames,
use the provided FrameMetricsState to hold previous values.
"""

# Your utilities
from typing import Dict, Optional, Tuple
import math

from .metrics import ema, velocity, rom_percent, distance_2d
from .pose_helpers import (
    elbow_angle, knee_angle, hip_angle, shoulder_angle, ankle_angle,
    torso_tilt, torso_tilt_signed,
    shoulder_mid, hip_mid,
    wrist_rel_shoulder_y, midfoot_x,
    knee_valgus, body_height_proxy,
    visibility_ok,
)



# -------------------------------
# Config: reference "top" and "bottom" targets for ROM mapping
# 0% = TOP/lockout, 100% = BOTTOM/deepest
# Tune these with your clips. They’re decent starting points.
# -------------------------------
ROM_REFS = {
    "pushup": {          # using elbow flexion (pick side later)
        "top_deg": 170.0,
        "bottom_deg": 60.0,
        "ema_alpha_rom": 0.30,
        "ema_alpha_vel": 0.50,
    },
    "squat": {           # using knee flexion
        "top_deg": 175.0,
        "bottom_deg": 70.0,
        "ema_alpha_rom": 0.30,
        "ema_alpha_vel": 0.50,
    },
    "ohp": {             # using wrist_rel_shoulder_y (unitless, torso-normalized)
        "top_val":  0.20,  # wrist ~0.20 torso-lengths above shoulder at lockout
        "bottom_val": -0.05, # wrist slightly below/near shoulder at bottom
        "ema_alpha_rom": 0.30,
        "ema_alpha_vel": 0.50,
    },
}

# Which side to use for single-signal ROM per exercise.
# You can switch to "min" or "avg" if you want to combine sides.
ROM_SIDE_STRATEGY = {
    "pushup": "min",    # use the *smaller* elbow angle (stricter)
    "squat":  "min",    # use the *smaller* knee angle
    "ohp":    "max",    # use the *higher* wrist height (better lockout proxy)
}

# -------------------------------
# State holder for EMA/velocity between frames
# -------------------------------
class FrameMetricsState:
    """
    Keep previous smoothed ROM% and last ROM% for velocity.
    One instance per exercise stream/camera.
    """
    def __init__(self):
        # Smoothed ROM per exercise
        self.rom_pushup_smoothed: Optional[float] = None
        self.rom_squat_smoothed: Optional[float] = None
        self.rom_ohp_smoothed: Optional[float] = None

        # Last ROM (pre-EMA) in case you want raw vel; we’ll do vel on smoothed
        self._last_rom_pushup: Optional[float] = None
        self._last_rom_squat: Optional[float] = None
        self._last_rom_ohp: Optional[float] = None

        # Last smoothed ROM to compute vel on smoothed values
        self._last_rom_pushup_smoothed: Optional[float] = None
        self._last_rom_squat_smoothed: Optional[float] = None
        self._last_rom_ohp_smoothed: Optional[float] = None


# -------------------------------
# Core extractor
# -------------------------------
def compute_frame_metrics(
    lm: Dict[str, Tuple[float, float, float, float]],
    dt: float,
    state: Optional[FrameMetricsState] = None
) -> Dict[str, Optional[float]]:
    """
    Build a metrics dict from landmarks in one pass.

    Notes:
      - Returns None for fields that can’t be computed (missing/low-visibility landmarks).
      - Uses EMA for ROM% when a state is provided.
      - Computes velocities on the *smoothed* ROM%.
    """
    out: Dict[str, Optional[float]] = {}

    # ---------------- Visibility ----------------
    # Minimal sets for robustness; tweak thresholds per your camera
    vis_ok_upper = visibility_ok(lm, [
        "L_SHOULDER","R_SHOULDER","L_ELBOW","R_ELBOW","L_WRIST","R_WRIST"
    ], thresh=0.5)
    vis_ok_lower = visibility_ok(lm, [
        "L_HIP","R_HIP","L_KNEE","R_KNEE","L_ANKLE","R_ANKLE"
    ], thresh=0.5)
    vis_ok_feet  = visibility_ok(lm, ["L_FOOT_INDEX","R_FOOT_INDEX"], thresh=0.5)

    out["vis_upper"] = 1.0 if vis_ok_upper else 0.0
    out["vis_lower"] = 1.0 if vis_ok_lower else 0.0
    out["vis_feet"]  = 1.0 if vis_ok_feet  else 0.0

    # ---------------- Raw angles & positions ----------------
    # Upper
    r_elbow = elbow_angle("right", lm) if vis_ok_upper else None
    l_elbow = elbow_angle("left", lm)  if vis_ok_upper else None
    r_shoulder_ang = shoulder_angle("right", lm) if vis_ok_upper and vis_ok_lower else None
    l_shoulder_ang = shoulder_angle("left", lm)  if vis_ok_upper and vis_ok_lower else None
    r_wrist_rel = wrist_rel_shoulder_y("right", lm) if vis_ok_upper and vis_ok_lower else None
    l_wrist_rel = wrist_rel_shoulder_y("left", lm)  if vis_ok_upper and vis_ok_lower else None

    # Lower
    r_knee = knee_angle("right", lm) if vis_ok_lower else None
    l_knee = knee_angle("left", lm)  if vis_ok_lower else None
    r_hip_ang = hip_angle("right", lm) if vis_ok_upper and vis_ok_lower else None
    l_hip_ang = hip_angle("left", lm)  if vis_ok_upper and vis_ok_lower else None
    r_ankle_ang = ankle_angle("right", lm) if vis_ok_lower else None
    l_ankle_ang = ankle_angle("left", lm)  if vis_ok_lower else None

    # Mids & tilt
    sm = shoulder_mid(lm) if vis_ok_upper else None
    hm = hip_mid(lm) if vis_ok_lower else None
    tilt = torso_tilt(lm) if (sm and hm) else None
    tilt_signed = torso_tilt_signed(lm) if (sm and hm) else None

    # Feet & balance
    midfoot = midfoot_x(lm) if vis_ok_feet else None
    r_valgus = knee_valgus("right", lm) if vis_ok_lower else None
    l_valgus = knee_valgus("left", lm)  if vis_ok_lower else None

    # Body size proxies
    torso_len = distance_2d(sm, hm) if (sm and hm) else None
    body_h = body_height_proxy(lm) if (sm and vis_ok_lower) else None

    # Populate raw outputs
    out.update({
        "r_elbow_deg": r_elbow, "l_elbow_deg": l_elbow,
        "r_knee_deg": r_knee,   "l_knee_deg": l_knee,
        "r_hip_deg": r_hip_ang, "l_hip_deg": l_hip_ang,
        "r_shoulder_deg": r_shoulder_ang, "l_shoulder_deg": l_shoulder_ang,
        "r_ankle_deg": r_ankle_ang, "l_ankle_deg": l_ankle_ang,

        "torso_tilt_deg": tilt,
        "torso_tilt_signed_deg": tilt_signed,

        "shoulder_mid_x": sm[0] if sm else None,
        "shoulder_mid_y": sm[1] if sm else None,
        "hip_mid_x": hm[0] if hm else None,
        "hip_mid_y": hm[1] if hm else None,

        "wrist_rel_y_r": r_wrist_rel,
        "wrist_rel_y_l": l_wrist_rel,

        "midfoot_x": midfoot,
        "knee_valgus_r": r_valgus,
        "knee_valgus_l": l_valgus,

        "torso_len": torso_len,
        "body_height": body_h,
    })

    # ---------------- Derived ROM% per exercise ----------------
    # PUSH-UP: use elbow angle(s) → ROM 0(top) to 100(bottom)
    pushup_rom = None
    if r_elbow is not None or l_elbow is not None:
        ref = ROM_REFS["pushup"]
        # side strategy
        candidates = [x for x in (r_elbow, l_elbow) if x is not None]
        if candidates:
            if ROM_SIDE_STRATEGY["pushup"] == "min":
                angle = min(candidates)
            elif ROM_SIDE_STRATEGY["pushup"] == "max":
                angle = max(candidates)
            else:
                angle = sum(candidates) / len(candidates)
            pushup_rom = rom_percent(angle, ref["top_deg"], ref["bottom_deg"])

    # SQUAT: knee angle(s) → ROM 0(top) to 100(bottom)
    squat_rom = None
    if r_knee is not None or l_knee is not None:
        ref = ROM_REFS["squat"]
        candidates = [x for x in (r_knee, l_knee) if x is not None]
        if candidates:
            if ROM_SIDE_STRATEGY["squat"] == "min":
                angle = min(candidates)
            elif ROM_SIDE_STRATEGY["squat"] == "max":
                angle = max(candidates)
            else:
                angle = sum(candidates) / len(candidates)
            squat_rom = rom_percent(angle, ref["top_deg"], ref["bottom_deg"])

    # OHP: wrist_rel_shoulder_y (unitless) → ROM 0(top) to 100(bottom)
    ohp_rom = None
    if r_wrist_rel is not None or l_wrist_rel is not None:
        ref = ROM_REFS["ohp"]
        candidates = [x for x in (r_wrist_rel, l_wrist_rel) if x is not None]
        if candidates:
            val = max(candidates) if ROM_SIDE_STRATEGY["ohp"] == "max" else \
                  min(candidates) if ROM_SIDE_STRATEGY["ohp"] == "min" else \
                  sum(candidates)/len(candidates)
            # Map with top_val/bottom_val (already signed/normalized)
            ohp_rom = rom_percent(val, ref["top_val"], ref["bottom_val"])

    out["rom_pushup"] = pushup_rom
    out["rom_squat"]  = squat_rom
    out["rom_ohp"]    = ohp_rom

    # ---------------- EMA and velocities (if state provided) ----------------
    if state is not None:
        # PUSH-UP
        if pushup_rom is not None:
            a_rom = ROM_REFS["pushup"]["ema_alpha_rom"]
            state.rom_pushup_smoothed = ema(state.rom_pushup_smoothed, pushup_rom, a_rom)
            out["rom_pushup_smooth"] = state.rom_pushup_smoothed
            # Velocity on smoothed ROM
            out["vel_pushup"] = velocity(state.rom_pushup_smoothed, state._last_rom_pushup_smoothed, dt) \
                                if state._last_rom_pushup_smoothed is not None else 0.0
            state._last_rom_pushup_smoothed = state.rom_pushup_smoothed
        else:
            out["rom_pushup_smooth"] = None
            out["vel_pushup"] = None

        # SQUAT
        if squat_rom is not None:
            a_rom = ROM_REFS["squat"]["ema_alpha_rom"]
            state.rom_squat_smoothed = ema(state.rom_squat_smoothed, squat_rom, a_rom)
            out["rom_squat_smooth"] = state.rom_squat_smoothed
            out["vel_squat"] = velocity(state.rom_squat_smoothed, state._last_rom_squat_smoothed, dt) \
                               if state._last_rom_squat_smoothed is not None else 0.0
            state._last_rom_squat_smoothed = state.rom_squat_smoothed
        else:
            out["rom_squat_smooth"] = None
            out["vel_squat"] = None

        # OHP
        if ohp_rom is not None:
            a_rom = ROM_REFS["ohp"]["ema_alpha_rom"]
            state.rom_ohp_smoothed = ema(state.rom_ohp_smoothed, ohp_rom, a_rom)
            out["rom_ohp_smooth"] = state.rom_ohp_smoothed
            out["vel_ohp"] = velocity(state.rom_ohp_smoothed, state._last_rom_ohp_smoothed, dt) \
                             if state._last_rom_ohp_smoothed is not None else 0.0
            state._last_rom_ohp_smoothed = state.rom_ohp_smoothed
        else:
            out["rom_ohp_smooth"] = None
            out["vel_ohp"] = None

    else:
        # no state: expose raw ROM and no velocity
        out["rom_pushup_smooth"] = pushup_rom
        out["rom_squat_smooth"]  = squat_rom
        out["rom_ohp_smooth"]    = ohp_rom
        out["vel_pushup"] = None
        out["vel_squat"]  = None
        out["vel_ohp"]    = None

    return out
