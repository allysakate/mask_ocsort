# MASKOCSORT
OC-Sort Tracker with feature descriptor from instance segmentation

## Environment Setup
1. Create our environment and install dependencies. See [Pytorch](https://pytorch.org/) and modify cudatoolkit version if needed.

        $ conda create --name mask_ocsort python=3.8
        $ conda activate mask_ocsort
        $ conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

2. Install [Detectron2](https://github.com/facebookresearch/detectron2) and other dependencies.

        $ python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
        $ python setup.py develop
        $ pip install -r requirements.txt
        $ pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

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
