# Implementation and Comparison of Pose Estimation Models on Unmanned Aerial Vehicle (UAV) Dataset
The paper presents a comparative study of the performance of three Pose Estimation models (OpenPose, MoveNet, and BlazePose) on the Unmanned Aerial Vehicle (UAV) dataset. In addition, we describe an approach to build on the OpenPose model through a fully connected network.

## Repository Structure
- `UAV Human Dataset/classes` contains  Unmanned Aerial Vehicle dataset used for our method

- `Blaze_Pose/` 
  - Contains `PoseEstimation_BlazePose.py` python files for Blazepose model implemented using mediapipe
  - `test_files.txt` contains test images path and `json_files.txt` contains json file path for corresponding images
  -  `Results/` contains annotated images obtained from running BlazePose model on test images dataset
  
- `MoveNet_Lightning/` 
  - Contains `lite-model_movenet_singlepose_lightning_3.tflite`, the lightweight MoveNet lightning model.
  - `MoveNet Lightning.ipynb`, the notebook implementing the MoveNet model and preciding keypoints on UAV test dataset.
  -  `test_files` contains the randomly generated files acting as the test dataset (generated using train_test_segregation)
  -  `output.txt` is used to store the complete path to test files.
  
 
