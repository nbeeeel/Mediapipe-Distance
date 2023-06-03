import mediapipe as mp
import cv2
import numpy as np
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

alpha = 0.3
previous_depth = 0.0

def apply_ema_filter(current_depth):
    global previous_depth
    filtered_depth = alpha * current_depth + (1 - alpha) * previous_depth
    previous_depth = filtered_depth  # Update the previous depth value
    return filtered_depth

def depth_to_distance(depth_value, depth_scale):
  return -1.0 / (depth_value * depth_scale)

cap = cv2.VideoCapture(0)
while cap.isOpened():
  ret,frame = cap.read()

#Grayscaling the image
  img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
  results = pose.process(img)

#check to see if bodylandmarks are being detected
  if results.pose_landmarks is not None:
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(img,results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    landmarks = []
    for landmark in results.pose_landmarks.landmark:
      landmarks.append((landmark.x, landmark.y, landmark.z))

    nose_landmark = landmarks[mp_pose.PoseLandmark.NOSE.value]
    nose_x, nose_y, nose_z = nose_landmark

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    distance = depth_to_distance(nose_z,1)
    distance = apply_ema_filter(distance)
    cv2.putText(img, "Depth in unit: " + str(np.format_float_positional(distance, precision=1)),(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3)
    cv2.imshow('ImgWindow',img)

  if cv2.waitKey(1) & 0xFF == ord('q'):
   cap.release()
   cv2.destroyAllWindows()