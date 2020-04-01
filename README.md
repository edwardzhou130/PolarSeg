# PolarNet
<p align="center">
        <img src="imgs/PC_vis.png" title="SemanticKITTI Point Cloud" width="48%"> <img src="imgs/predict_vis.png" title="PolarSeg Prediction" width="48%">
        <em>Point cloud visualization of SemanticKITTI dataset(left) and the prediction result of PolarNet(right).</em>
</p>

This is the official implmentation for CVPR 2020 paper ["PolarNet: An Improved Grid Representation for Online LiDAR Point Clouds Semantic Segmentation"](https://arxiv.org/abs/2003.14032).

## What is PolarNet?

PolarNet is a lightweight neural network that aims to provide near-real-time online semantic segmentation for a single LiDAR scan. Unlike existing methods that require KNN to build a graph and/or 3D/graph convolution, we achieve fast inference speed by avoiding both of them. As shown below, we quantize points into grids using their polar coordinations. We then learn a fixed-length representation for each grid and feed them to a 2D neural network to produce point segmentation results.

<p align="center">
        <img src="imgs/overview.png" width="90%"> 
</p>

## Prepare dataset and environment

1, Install the following dependencies using `conda install --file requirements.txt`.
* numpy
* pytorch
* tqdm
* yaml
* numba
* torch_scatter

2, Download Velodyne point clouds and label data in SemanticKITTI dataset [here](http://www.semantic-kitti.org/dataset.html#overview).

3, Extract everything into the same folder. The folder structure inside the zip files of label data matches the folder structure of the LiDAR point cloud data.

4, Data file structure should look like this:

```shell
./
├── train.py
├── ...
└── data/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	# Unzip from KITTI Odometry Benchmark Velodyne point clouds.
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 	# Unzip from SemanticKITTI label data.
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── ...
        └── 21/
	    └── ...
```
## Training

```shell
python train.py
```

## Evaluate our pretrained model

```shell
python test_pretrain.py
```
Result will be stored in `./out` folder. Test performance can be evaluated by uploading label results onto the SemanticKITTI competition website [here](https://competitions.codalab.org/competitions/20331).

Remember to shift label number back to the original dataset format before submitting! Instruction can be found in [semantic-kitti-api repo](https://github.com/PRBonn/semantic-kitti-api).

## Citation
Please cite our paper if this code benefits your reseaarch:
```
@InProceedings{Zhang_2020_PolarNet,
author = {Yang Zhang and Zixiang Zhou and Philip David and Xiangyu Yue and Zerong Xi and Hassan Foroosh},
title = {PolarNet: An Improved Grid Representation for Online LiDAR Point Clouds Semantic Segmentation},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
Year = {2020},
}
```

