#REFERENCE: https://github.com/google/mediapipe

#import libraries
import cv2
import mediapipe as mp
import numpy as np
import json
import time

# Initialize mediapipe drawing class - to draw the landmarks points.
mp_drawing = mp.solutions.drawing_utils

# Initialize mediapipe pose class
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# For static images:
IMAGES = []
JSONS = []
image_files = 'test_files.txt'
json_files = 'json_files.txt'

#Read the test image and corresponding json file containing file paths into a list
with open(image_files, 'r') as file:
    lines = file.readlines()
IMAGES = [line.strip() for line in lines]

Number_of_test_images = len(IMAGES)

with open(json_files, 'r') as file:
    lines = file.readlines()
JSONS = [line.strip() for line in lines]


preds = []

# Define common keypoints mapping
common_keypoints = {'nose':0,
'leftEye':2,
'rightEye':5,
'leftEar':7,
'rightEar':8,
'leftShoulder':11,
'rightShoulder':12,
'leftElbow':13,
'rightElbow':14,
'leftWrist':15,
'rightWrist':16,
'leftHip':23,
'rightHip':24,
'leftKnee':25,
'rightKnee':26,
'leftAnkle':27,
'rightAnkle':28}

# Initialize mAP value to zero
mAP_Result = 0

# Find ground_truth_points for given images
def load_ground_truth_points(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    ground_truth_points = []
    # Get x,y co-ordinates from annotations
    for completion in data['completions']:
        for item in completion['result']:
            if item['type'] == 'keypointlabels':
                label = item['value']['keypointlabels'][0]
                x = item['value']['x']
                y = item['value']['y']
                ground_truth_points.append((label, x, y))
    return ground_truth_points

# Calculate mAP(mean Average Precision)
def mAP(true_points, pred_points, body_pts, thresh=5):
    # Initialize the precision and recall arrays
    precision = np.zeros(body_pts)
    recall = np.zeros(body_pts)

    # Iterate over each body part
    for i in range(body_pts):
        # Convert true points to (x, y) coordinates
        true_coords = np.array(true_points[i][1:3], dtype=np.float64)
        pred_coords = np.array(pred_points[i][1:3], dtype=np.float64)

        # Calculate the Euclidean distance between the predicted and ground truth coordinates for the body part
        dist = np.linalg.norm(pred_coords - true_coords)

        # Assign a label of 1 (true positive) or 0 (false positive) based on whether the distance is below or above the threshold
        label = int(dist <= thresh)

        # Update the precision and recall arrays
        if label == 1:
            precision[i] = 1
            recall[i] = 1
        else:
            precision[i] = 0
            recall[i] = 0

    # Calculate the cumulative precision and recall arrays
    cum_precision = np.cumsum(precision) / (np.arange(body_pts) + 1)
    cum_recall = np.cumsum(recall) / body_pts

    # Calculate the average precision
    ap = np.sum((cum_recall[1:] - cum_recall[:-1]) * cum_precision[1:])

    return ap

with mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5) as pose:

    for index, (file1, file2) in enumerate(zip(JSONS, IMAGES)):
        json_file_path = file1
        truth_points = load_ground_truth_points(json_file_path)
        image = cv2.imread(file2)
        image_height, image_width, _ = image.shape

        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw pose landmarks on the image.
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite(r'annotated '+ str(index) +'.png', annotated_image)

        # Find predicted keypoints
        if results.pose_landmarks:
            pred_keypoints = []
            for idx, lmk in enumerate(results.pose_landmarks.landmark):
                # Convert normalized landmark coordinates to image coordinates
                x = lmk.x * 100
                y = lmk.y * 100

                label = 'Keypoint {}'.format(idx)
                pred_keypoints.append((idx, x, y))
        else:
            # If no pose landmarks are detected, set all keypoints to zero
            pred_keypoints = [(idx, 0, 0) for idx in range(33)]

        # Sort the keypoints based on their indices
        pred_keypoints.sort(key=lambda x: x[0])

        # Extract the (x, y) coordinates and labels
        pred_keypoints_coords = np.array([(x, y) for _, x, y in pred_keypoints])
        pred_keypoints_labels = [label for label, _, _ in pred_keypoints]

        # Append the flattened keypoints and labels to the 'preds' list
        preds.append(np.concatenate((pred_keypoints_coords.flatten(), pred_keypoints_labels)))

        final_predicted_keypoints=[]

        # Get common keypoints of predicted and truth points
        for key in truth_points:
            label = common_keypoints[key[0]]
            final_predicted_keypoints.append(pred_keypoints[label])

        mAP_Result = mAP_Result + mAP(truth_points, final_predicted_keypoints, len(truth_points))

    # Find mAP for the test dataset
    Avg_mAP = mAP_Result / Number_of_test_images
    print("Average_mAP", Avg_mAP)



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