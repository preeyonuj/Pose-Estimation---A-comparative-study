# Implementation and Comparison of Pose Estimation Models on Unmanned Aerial Vehicle (UAV) Dataset
The paper presents a comparative study of the performance of three Pose Estimation models (OpenPose, MoveNet, and BlazePose) on the Unmanned Aerial Vehicle (UAV) dataset. In addition, we describe an approach to build on the OpenPose model through a fully connected network.

## Repository Structure
- `UAV Human Dataset/classes` contains  Unmanned Aerial Vehicle dataset used for our method

- `Blaze_Pose/` 
  - Contains `PoseEstimation_BlazePose.py` python files for Blazepose model implemented using mediapipe.
  - `test_files.txt` contains test images dataset path (./classes/) and `json_files.txt` contains json file path (./classes/) for corresponding images.
  -  `Results/` contains annotated images obtained from running BlazePose model on test images dataset.
  
- `MoveNet_Lightning/` 
  - Contains `lite-model_movenet_singlepose_lightning_3.tflite`, the lightweight MoveNet lightning model.
  - The notebook to be run: `MoveNet Lightning.ipynb`, is the  the MoveNet modelimplementation of the MoveNet model and predicts keypoints on UAV test dataset.
  -  `test_files` contains the randomly generated files acting as the test dataset (generated using train_test_segregation).
  -  `output.txt` is used to store the complete path to test files (this is generated by the code).

 - `OpenPose_base` 
    - Contains OpenPose notebooks for processing the train and test files along with the required scripts for model, and pose.
    - Please download the body_pose_model.pth weights from the link https://drive.google.com/drive/folders/1JsvI4M4ZTg98fmnCZLFM-3TeovnCRElG and place it inside the OpenPose_base folder.
    - Run both notebooks: OpenPose_baseline.ipynb and OpenPose_Train_points_generation.ipynb.
    - It will generate two files: test_coords.csv and train_coords.csv, which will be used by the finetuned model notebook to generate the results.

- `Open Pose Finetuned/` 
  - Run Finetuned_OpenPose.ipynb to get the final output of the finetuned OpenPose model.
