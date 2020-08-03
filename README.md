# 3DSSD-pytorch
 3DSSD's implementation with Pytorch

This repository contains a PyTorch implementation of [3DSSD](https://github.com/Jia-Research-Lab/3DSSD) on the KITTI benchmark.

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
