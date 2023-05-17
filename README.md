# Implementation and Comparison of OpenPose Model on UAV Dataset
The project focuses on implementing the OpenPose model and fine-tuning it using the UAV-Human dataset. The goal is to compare the performance of the fine tuned OpenPose model with two other pose estimation algorithms, namely  BlazePose and MoveNet, using the same dataset.

## Repository Structure
- `Blaze_Pose/` 
  - Contains `PoseEstimation_BlazePose.py` python files for Blazepose model implemented using mediapipe
  - `test_files.txt` contains test images path and `json_files.txt` contains json file path for corresponding images
  -  `Results/` contains annotated images obtained from running BlazePose model 
- `UAV Human Dataset/classes` contains  Unmanned Aerial Vehicle dataset used for our method
  
 
