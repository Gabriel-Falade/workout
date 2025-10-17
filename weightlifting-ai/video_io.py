import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found"); raise SystemExit

while True:
    ok, frame = cap.read()
    if not ok: break
    mirrored_frame = cv.flip(frame, 1)
    cv.imshow("Webcam", mirrored_frame)
    if cv.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv.destroyAllWindows()
