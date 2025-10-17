import cv2 as cv
import mediapipe as mp
import numpy as np
import time

# ------------------------------
# Globals
# ------------------------------
counter = 0
stage = None

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ------------------------------
# Geometry helpers
# ------------------------------
def calculate_angle(a, b, c):
    """Angle at b formed by a–b–c, in degrees [0,180]."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)

# ------------------------------
# Overlay: bottom-right info box
# ------------------------------
def draw_info_box(img, lines, padding=10, line_height=22):
    """
    Draw a solid info box with calculation lines in the bottom-right corner.
    `lines` is a list of strings.
    """
    if img is None or len(lines) == 0:
        return img

    h, w = img.shape[:2]
    font = cv.FONT_HERSHEY_COMPLEX
    font_scale = 0.5
    thickness = 1

    # Compute box size
    text_sizes = [cv.getTextSize(s, font, font_scale, thickness)[0] for s in lines]
    text_w = max(ts[0] for ts in text_sizes) if text_sizes else 0
    text_h = line_height * len(lines)
    box_w = text_w + padding * 2
    box_h = text_h + padding * 2

    # Bottom-right anchor
    x1 = max(0, w - box_w - 10)
    y1 = max(0, h - box_h - 10)
    x2 = x1 + box_w
    y2 = y1 + box_h

    # Draw filled rectangle (semi-opaque look by blending)
    overlay = img.copy()
    cv.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    alpha = 0.65
    img = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Draw text lines
    y = y1 + padding + 15
    for s in lines:
        cv.putText(img, s, (x1 + padding, y), font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)
        y += line_height

    return img

# ------------------------------
# Main
# ------------------------------
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found")
    raise SystemExit

# For velocity/debug timing if you want to extend later
prev_time = time.time()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Process
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # Extract landmark-based angles
        r_angle = None
        l_angle = None

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Get coords (normalized 0..1)
            r_shoulder = (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
            r_elbow    = (lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y)
            r_wrist    = (lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y)

            l_shoulder = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
            l_elbow    = (lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y)
            l_wrist    = (lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y)

            # Angles
            r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

            # Visualize elbow angles near joints (use actual frame dims)
            H, W = image.shape[:2]
            def denorm(p):  # normalized -> pixel coords
                return (int(p[0] * W), int(p[1] * H))

            cv.putText(image, f"{r_angle:.1f}",
                       denorm(r_elbow),
                       cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(image, f"{l_angle:.1f}",
                       denorm(l_elbow),
                       cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)

            # ------------------------------
            # Simple checker logic (Shoulder Press-style)
            # DOWN when both elbows ~90°; UP when both extended
            # ------------------------------
            global stage, counter  # modify the globals declared at top
            if r_angle is not None and l_angle is not None:
                if 80 <= r_angle <= 100 and 80 <= l_angle <= 100:
                    stage = "DOWN"
                elif r_angle >= 160 and l_angle >= 160 and stage == "DOWN":
                    stage = "UP"
                    counter += 1
                    # print(counter)  # optional

        # Draw pose skeleton
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # Mirror for user-friendly view
        image = cv.flip(image, 1)

        # Top-left small box: reps & stage (kept from your original UI)
        cv.rectangle(image, (0, 0), (230, 74), (245, 117, 16), -1)
        cv.putText(image, 'REPS', (15, 14), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(image, str(counter), (10, 62), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

        cv.putText(image, 'STAGE', (88, 14), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(image, (stage or "--"), (80, 62), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

        # Bottom-right info box: calculations
        info_lines = [
            f"Right Elbow: {r_angle:.1f} deg" if r_angle is not None else "Right Elbow: --",
            f"Left  Elbow: {l_angle:.1f} deg" if l_angle is not None else "Left  Elbow: --",
            f"Counter: {counter}",
            f"Mode: Shoulder Press Checker"
        ]
        image = draw_info_box(image, info_lines)

        cv.imshow("Webcam", image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
