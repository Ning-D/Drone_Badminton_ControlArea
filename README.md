# Drone_Badminton_ControlArea


## Requirements
- python 3.8
- When creating a new virtual environment, run `pip install -r requirements.txt`


## Usage
- First, check and set the folder structure of input videos.
- Pretrained weights yolov5x.pt can be download from: https://drive.google.com/file/d/172_phnHIz5MmGQ_EORry1rEng3zH-3yB/view?usp=share_link
- For train and test, please run `python main.py
- For control area visualization, please run `python save.py --checkpoint_path ./epo30_lr1e-06_w0_A0.5_B0.5_G3_K0.03/model_X.pth`

