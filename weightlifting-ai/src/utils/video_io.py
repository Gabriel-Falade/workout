import cv2 as cv
import mediapipe as mp
import numpy as np

counter = 0
stage = None


# Gives us drawing utilities 
mp_drawing = mp.solutions.drawing_utils
# Import pose estimation model
mp_pose = mp.solutions.pose

# Open my webcam
cap = cv.VideoCapture(0)

# angle calculations 
def calculate_angle(a,b,c): #first, mid, and endpoint
    a = np.array(a) #first
    b = np.array(b) #mid
    c = np.array(c) #end

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180:   #max angle we want is 180
        angle = 360 - angle

    return angle

def curl_logic():
    if (r_angle > 170) :
        stage = "down"
    if (r_angle > 30) and stage == 'down':
            stage = 'up'
            counter += 1
            print(counter)


def shoulder_press():
    if (r_angle > 80 and r_angle < 100) and (l_angle > 80 and l_angle < 100):
        stage = "down"
    if (r_angle > 160 and l_angle > 160) and stage == 'down':
        stage = 'up'
        counter += 1
        print(counter)

# Check to see if webcam was succefully opened
if not cap.isOpened():
    print("Camera not found"); raise SystemExit

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        # Read the first frame to confirm reading
        ret, frame = cap.read() # ret -> indicates if frame was successfully captured (boolean)
        if not ret: break
        
        # Detect stuff and reader 
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False # saves memory 

        # Makes detection 
        results = pose.process(image)
        # Convering color back
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # Extracting landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            # print(landmarks)

            # Get coordinates 
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            #Calculate angle
            r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)


            #visiulization
            cv.putText(image, str(r_angle),
                       tuple(np.multiply(r_elbow, [640, 480]).astype(int)),  #placement of angle
                             cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(image, str(l_angle),
                       tuple(np.multiply(l_elbow, [640, 480]).astype(int)),  #placement of angle
                             cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)


            # Curl logic 
            if (r_angle > 80 and r_angle < 100) and (l_angle > 80 and l_angle < 100):
                stage = "down"
            if (r_angle > 160 and l_angle > 160) and stage == 'down':
                stage = 'up'
                counter += 1
                print(counter)


        except:
            pass

        

        # Render detection 
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), #color at dif dots 
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) # color of dif lines

        #Flips the frame
        mirrored_image = cv.flip(image, 1)


        # Rendering curl logic 
        cv.rectangle(mirrored_image, (0,0), (225,73), (245, 117, 16), -1)

        cv.putText(mirrored_image, 'REPS', (15,12),
                   cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
        cv.putText(mirrored_image, str(counter),
                   (10,60),
                   cv.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2, cv.LINE_AA)
        
        # Rendering stage
        cv.putText(mirrored_image, 'STAGE', (65,12),
                   cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
        cv.putText(mirrored_image, stage,
                   (60,60),
                   cv.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2, cv.LINE_AA)
        

        cv.imshow("Webcam", mirrored_image)
        if cv.waitKey(1) & 0xFF == ord('q'): break

print()

# Release the video capture object
cap.release()
cv.destroyAllWindows()
