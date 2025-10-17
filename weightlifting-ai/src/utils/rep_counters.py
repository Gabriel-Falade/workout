import numpy as np
import cv2 as cv


# calculating angle 
def calculate_angle(a,b,c): #first, mid, and endpoint
    a = np.array(a) #first
    b = np.array(b) #mid
    c = np.array(c) #end

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180:   #max angle we want is 180
        angle = 360 - angle

    return angle



def shoulder_press_count():
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