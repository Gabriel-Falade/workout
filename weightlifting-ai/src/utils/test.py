# src/utils/test.py
import os, sys, time
from typing import List, Optional, Dict, Tuple
import cv2 as cv
import mediapipe as mp

# --- Make 'src' importable when running from src/utils ---
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))  # .. = src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Project imports
from analysis.pose_helpers import mp_results_to_dict
from analysis.frame_metrics import compute_frame_metrics, FrameMetricsState

# Registry: exercise -> (module_path, class_name, needs_landmarks)
EXERCISE_REGISTRY: Dict[str, Tuple[str, str, bool]] = {
    # already built in your repo
    "pushup":    ("exercises.pushup",        "PushUpDetector",      False),
    "squat":     ("exercises.squat",         "SquatDetector",       False),
    "lunge":     ("exercises.lunge",         "LungeDetector",       False),
    "situp":     ("exercises.situp",         "SitupDetector",       False),

    # new batch you added
    "plank":     ("exercises.plank",         "PlankMonitor",        False),
    "jacks":     ("exercises.jumping_jacks", "JumpingJacksDetector", True),
    "highknees": ("exercises.high_knees",    "HighKneesCounter",     True),
    "donkey":    ("exercises.donkey_kicks",  "DonkeyKicksDetector",  True),  # per-side default inside class
    "bridge":    ("exercises.glute_bridge",  "GluteBridgeDetector",  False),
    "jumpsquat": ("exercises.jump_squat",    "JumpSquatDetector",    True),
    "burpee":    ("exercises.burpee",        "BurpeeDetector",       True),
}

# ------------------------------
# Camera helper: try multiple indices/backends on Windows
# ------------------------------
def open_camera():
    indices = [0, 1, 2]
    backends = [cv.CAP_DSHOW, cv.CAP_MSMF, cv.CAP_ANY]
    for i in indices:
        for be in backends:
            cap = cv.VideoCapture(i, be)
            if cap.isOpened():
                return cap
            cap.release()
    return None

# ------------------------------
# UI helpers
# ------------------------------
def draw_info_box(img, lines: List[str], padding=10, line_height=22):
    """Bottom-right semi-opaque box with lines of text."""
    if img is None or not lines:
        return img
    h, w = img.shape[:2]
    font = cv.FONT_HERSHEY_COMPLEX
    font_scale = 0.5
    thickness = 1

    sizes = [cv.getTextSize(s, font, font_scale, thickness)[0] for s in lines]
    text_w = max((sz[0] for sz in sizes), default=0)
    box_w = text_w + padding * 2
    box_h = line_height * len(lines) + padding * 2

    x1 = max(0, w - box_w - 10)
    y1 = max(0, h - box_h - 10)
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = img.copy()
    cv.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    img = cv.addWeighted(overlay, 0.65, img, 0.35, 0)

    y = y1 + padding + 15
    for s in lines:
        cv.putText(img, s, (x1 + padding, y), font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)
        y += line_height
    return img

def draw_toast(img, lines: List[str], color=(60, 180, 75), state: Optional[Dict]=None):
    """
    Bottom-center toast shown while state['until_ts'] is in the future.
    state must contain: {'until_ts': float}
    """
    if img is None or not lines or state is None:
        return img
    if time.time() > state.get("until_ts", 0.0):
        return img

    h, w = img.shape[:2]
    font = cv.FONT_HERSHEY_COMPLEX
    fs, th = 0.7, 2

    sizes = [cv.getTextSize(s, font, fs, th)[0] for s in lines]
    text_w = max((sz[0] for sz in sizes), default=0)
    line_h = 28
    pad_x, pad_y = 16, 14

    box_w = text_w + pad_x * 2
    box_h = line_h * len(lines) + pad_y * 2

    x1 = max(10, (w - box_w) // 2)
    y2 = h - 10
    y1 = y2 - box_h

    overlay = img.copy()
    cv.rectangle(overlay, (x1, y1), (x1 + box_w, y2), color, -1)
    img = cv.addWeighted(overlay, 0.7, img, 0.3, 0)

    y = y1 + pad_y + 18
    for s in lines:
        cv.putText(img, s, (x1 + pad_x, y), font, fs, (255, 255, 255), th, cv.LINE_AA)
        y += line_h

    return img

def arm_toast(state: Dict, show_ms: int = 1600):
    state["until_ts"] = time.time() + (show_ms / 1000.0)

def class_to_color(rep_class: str, counted: bool):
    """Color mapping for toast."""
    if counted:
        return (60, 180, 75)       # green
    rc = (40, 40, 220)            # red default
    if rep_class and "warn" in rep_class.lower():
        rc = (0, 215, 255)        # yellow
    return rc

# ------------------------------
# Human selection
# ------------------------------
def choose_exercise() -> Tuple[str, bool, object]:
    print("\n=== Select an exercise ===")
    names = sorted(EXERCISE_REGISTRY.keys())
    for idx, name in enumerate(names, 1):
        print(f"{idx}. {name}")
    choice = input("Type name or number: ").strip().lower()

    key = None
    if choice.isdigit():
        i = int(choice) - 1
        if 0 <= i < len(names):
            key = names[i]
    else:
        if choice in EXERCISE_REGISTRY:
            key = choice

    if key is None:
        print("Invalid choice. Defaulting to 'pushup'.")
        key = "pushup"

    module_path, class_name, needs_lm = EXERCISE_REGISTRY[key]
    mod = __import__(module_path, fromlist=[class_name])
    Detector = getattr(mod, class_name)
    # Donkey kicks default side can be changed here, e.g., Detector(side="left")
    detector = Detector()
    print(f"Running: {key}")
    return key, needs_lm, detector

# ------------------------------
# Main
# ------------------------------
def main():
    exercise_key, needs_lm, detector = choose_exercise()

    cap = open_camera()
    if cap is None:
        print("ERROR: Could not open any camera (tried indices 0..2 with DirectShow/MSMF).")
        return

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    fm_state = FrameMetricsState()

    # toast/feedback state
    toast = {"until_ts": 0.0, "lines": [], "color": (60, 180, 75)}

    fps_ema = None
    prev = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARN: Failed to read frame.")
                break

            # Pose
            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)

            image = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
            image.flags.writeable = True

            # Landmarks -> dict (x,y,*,visibility) and raw list for some detectors
            lm = mp_results_to_dict(results)

            # Timing / FPS
            now = time.time()
            dt = max(1e-3, now - prev)
            prev = now
            fps_inst = 1.0 / dt
            fps_ema = fps_inst if fps_ema is None else (0.2 * fps_inst + 0.8 * fps_ema)

            # Frame metrics (angles, tilt, ROMs, etc.)
            fm = compute_frame_metrics(lm, dt, fm_state)

            # Draw skeleton
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            # Detector update
            if needs_lm:
                rep_event, live = detector.update(fm, lm, now_s=now)
            else:
                rep_event, live = detector.update(fm, now_s=now)

            # Mirror for user-friendly view
            image = cv.flip(image, 1)

            # Top-left box: Attempts & Reps & Stage (when available)
            cv.rectangle(image, (0, 0), (360, 90), (245, 117, 16), -1)
            cv.putText(image, 'ATTEMPTS', (12, 16), cv.FONT_HERSHEY_COMPLEX, 0.48, (0,0,0), 1, cv.LINE_AA)
            cv.putText(image, str(live.get("attempt_count","-")), (10, 78), cv.FONT_HERSHEY_COMPLEX, 1.8, (255,255,255), 2, cv.LINE_AA)

            cv.putText(image, 'REPS', (140, 16), cv.FONT_HERSHEY_COMPLEX, 0.48, (0,0,0), 1, cv.LINE_AA)
            cv.putText(image, str(live.get("rep_count","-")), (136, 78), cv.FONT_HERSHEY_COMPLEX, 1.8, (255,255,255), 2, cv.LINE_AA)

            cv.putText(image, 'STAGE', (235, 16), cv.FONT_HERSHEY_COMPLEX, 0.48, (0,0,0), 1, cv.LINE_AA)
            cv.putText(image, str(live.get("stage") or "--"), (231, 78), cv.FONT_HERSHEY_COMPLEX, 1.8, (255,255,255), 2, cv.LINE_AA)

            # Bottom-right info box
            if exercise_key == "plank":
                held_s = live.get("held_s", 0.0) or 0.0
                info_lines = [
                    f"Held: {held_s:4.1f}s",
                    f"FPS:  {fps_ema:4.1f}" if fps_ema is not None else "FPS:  --",
                    f"Mode: plank",
                ]
            else:
                rom = live.get("rom")
                vel = live.get("vel")
                info_lines = [
                    f"ROM:  {rom:5.1f} %" if isinstance(rom,(int,float)) else "ROM:  --",
                    f"Vel:  {vel:5.1f} %/s" if isinstance(vel,(int,float)) else "Vel:  --",
                    f"FPS:  {fps_ema:4.1f}" if fps_ema is not None else "FPS:  --",
                    f"Mode: {exercise_key}",
                ]
            image = draw_info_box(image, info_lines)

            # Rep feedback toast
            if rep_event:
                counted = bool(rep_event.get("counted", True))
                rep_cls = rep_event.get("class", "good")
                header = f"{exercise_key.title()} - {'COUNTED' if counted else 'NOT COUNTED'} [{rep_cls.upper()}]"
                lines = [header] + (rep_event.get("cues", [])[:2])
                toast["lines"] = lines
                toast["color"] = class_to_color(rep_cls, counted)
                arm_toast(toast, show_ms=1600)

            # Draw toast if armed
            image = draw_toast(image, toast.get("lines", []), color=toast.get("color", (60,180,75)), state=toast)

            cv.imshow("Webcam", image)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
