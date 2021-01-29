# 3DSSD-pytorch

## Important Note
This implementation has some bugs. The bugs are fixed in https://github.com/zye1996/3DSSD-torch. Thanks for his efforts in the debugging process since the debugging is the much more difficult that write the bugs.

I have implemented the 3DSSD in the [OpenPCDet](https://github.com/open-mmlab) framework. Thanks for the mmlab to provide such a good framework for 3D object detection!!!

I have put new implementation into a new [Repositories](https://github.com/qiqihaer/3DSSD-pytorch-openPCDet).
The eval performance is very satisfactory:
```
Car AP@0.70, 0.70, 0.70:
bbox AP:96.5468, 90.0235, 89.4066
bev  AP:90.3444, 88.0784, 86.0698
3d   AP:89.2219, 78.8593, 77.5890
aos  AP:96.52, 89.95, 89.25
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.2011, 95.0305, 92.6650
bev  AP:93.2919, 89.1952, 88.1910
3d   AP:91.4331, 82.2283, 77.8059
aos  AP:98.18, 94.93, 92.49
Car AP@0.70, 0.50, 0.50:
bbox AP:96.5468, 90.0235, 89.4066
bev  AP:96.6237, 90.1257, 89.6772
3d   AP:96.5594, 90.0998, 89.6259
aos  AP:96.52, 89.95, 89.25
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.2011, 95.0305, 92.6650
bev  AP:98.3041, 95.4983, 95.0182
3d   AP:98.2703, 95.3970, 94.8667
aos  AP:98.18, 94.93, 92.49
```


## Introduction
3DSSD's implementation with Pytorch

This repository contains a PyTorch implementation of [3DSSD](https://github.com/Jia-Research-Lab/3DSSD) on the KITTI benchmark.

There are several characteristics to make you easy to understand and modify the code:
1. I keep the name of the folders. files and the fucntions the same as the [official code](https://github.com/Jia-Research-Lab/3DSSD) as much as possible. 
2. The "Trainner" in the lib/core/trainner.py draws on the code style of the [PCDet](https://github.com/open-mmlab/OpenPCDet).
3. I borrow the visualization code with the MeshLab from the [VoteNet](https://github.com/facebookresearch/votenet).

If you want to use the F-FPS or the Dilated-Ball-Query, you can just use the files in lib/pointnet2.

## Preparation

1. Clone this repository

2. Install the Python dependencies.

```
pip install -r requirements.txt
```

3. Install python functions. the functions are partly borrowed from the Pointnet2 in [PointRCNN](https://github.com/sshaoshuai/PointRCNN). The F-FPS and dilated-ball-query are implemented by myself.

```
cd lib/pointnet2
python setup.py install
```

4. Prepare data with according to the "Data Preparation" in the [3DSSD](https://github.com/Jia-Research-Lab/3DSSD)

## Train a Model

```
python lib/core/trainer.py --cfg configs/kitti/3dssd/3dssd.yaml
```

The trainning log and tensorboard log are saved into output dir


