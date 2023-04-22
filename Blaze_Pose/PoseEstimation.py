import cv2
import mediapipe as mp
import time

# Initialize mediapipe drawing class - to draw the landmarks points.
mp_drawing = mp.solutions.drawing_utils

# Initialize mediapipe pose class
mp_pose = mp.solutions.pose

mp_holistic = mp.solutions.holistic

# For static images:
IMAGE_FILES = [
               'P001S05G20B00H00UC071000LC031000A121R0_09051655_006.jpg',
               'P001S05G20B00H00UC071000LC031000A129R0_09051655_105.jpg',
               'P002S00G10B50H30UC062000LC092000A037R0_08250945_099.jpg',
               'P002S02G11B00H00UC062001LC032001A034R0_08310854_002.jpg',
               'P002S04G10B00H10UC062000LC092000A055R0_09071427_060.jpg',
               'P002S07G10B40H00UC062000LC092000A105R0_09071427_019.jpg',
               'P003S00G10B50H30UC012000LC021000A004R0_08251016_016.jpg',
               'P076S04G10B40H30UC012000LC021000A064R0_09161703_053.jpg',
               'P115S29G11B10H10UC041021LC021021A093R0_10021556_121.jpg',
               'P000S00G10B10H10UC022000LC021000A012R0_08241716_018.jpg'
               ]
with mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5) as pose:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape

        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw pose landmarks on the image.
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite(r'annotated '+ str(idx) +'.png', annotated_image)


# For webcam input:
cap = cv2.VideoCapture(0)

#For Video input:
#cap = cv2.VideoCapture("filename")

prevTime = 0
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
    cv2.imshow('BlazePose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()


