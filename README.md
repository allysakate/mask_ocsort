# MASKOCSORT
OC-Sort Tracker with feature descriptor from instance segmentation

## Environment Setup
1. Create our environment and install dependencies. See [Pytorch](https://pytorch.org/) and modify cudatoolkit version if needed.

        $ conda create --name mask_ocsort python=3.8
        $ conda activate mask_ocsort
        $ conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge

2. Install [Detectron2](https://github.com/facebookresearch/detectron2) and other dependencies.

        $ python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
        $ pip install -r requirements.txt

3. Install pre-commit and run on all files

        $ pre-commit install
        $ pre-commit run --all-files

## Tracker
1. Modify configs/demo_config.yaml
2. Run

        $ python demo.py

## Calibration based on Tracker
1. Modify configs/calib_config.yaml
2. Run

        $ python calibration.py


## References
1. https://github.com/noahcao/OC_SORT
2. https://github.com/facebookresearch/detectron2
3. https://github.com/WongKinYiu/yolov7
4. https://github.com/nwojke/deep_sort


## Citation
If you find this work useful, please consider to cite our paper:

[1] A. K. Brillantes, E. Sybingco, A. Bandala, R. K. Billones, A. Fillone, and E. Dadios, “Vehicle Tracking in Low Frame Rate Scenes using Instance Segmentation,” in 2022 IEEE 14th International Conference on Humanoid, Nanotechnology, Information Technology, Communication and Control, Environment, and Management (HNICEM), Dec. 2022, pp. 1–5. doi: 10.1109/HNICEM57413.2022.10109390.

[2] A. K. Brillantes, E. Sybingco, R. K. Billones, A. Bandala, A. Fillone, and E. Dadios, “Observation-Centric with Appearance Metric for Computer Vision-Based Vehicle Counting,” JAIT, vol. 14, no. 6, pp. 1261–1272, 2023, doi: 10.12720/jait.14.6.1261-1272.
