# Self-Supervised Pretraining of 3D Features on any Point-Cloud
This code provides a PyTorch implementation and pretrained models for **DepthContrast**, as described in the paper [Self-Supervised Pretraining of 3D Features on any Point-Cloud](http://arxiv.org/abs/2101.02691).

<div align="center">
  <img width="100%" alt="DepthContrast Pipeline" src="imgs/method.jpg">
</div>

DepthContrast is an easy to implement self-supervised method that works across model architectures, input data formats, indoor/outdoor 3D, single/multi-view 3D data.
Similarly to 2D contrastive approaches, DepthContrast learns representations by comparing transformations of a 3D pointcloud/voxel. It does not require any multi-view information between frames, such as point-to-point correspondances. It makes our framework generalize to any 3D pointcloud or voxel input. 
DepthContrast pretrains high capacity models for 3D recognition tasks, and leverages large-scale 3D data. It shows state-of-the-art performance on detection and segmentation benchmarks, outperforming all prior work on detection.

# Model Zoo

We release our PointNet++ and MinkowskiEngine UNet models pretrained with DepthContrast with the hope that other researchers might also benefit from these pretrained backbones. Due to license issue, models pretrained on Waymo cannot be released. For PointnetMSG and Spconv-UNet models, we encourage the researchers to train by themselves using the provided script.

We first provide PointNet++ models with different sizes.
| network | epochs | batch-size | ScanNet Det with VoteNet | url | args |
|-------------------|---------------------|---------------------|--------------------|--------------------|--------------------|
| PointNet++-1x | 150 | 1024 | 61.9 | [model](https://dl.fbaipublicfiles.com/DepthContrast/pointnet1x/checkpoint-ep150.pth.tar) | [config](./configs/point_within_format.yaml) |
| PointNet++-2x | 200 | 1024 | 63.3 | [model](https://dl.fbaipublicfiles.com/DepthContrast/pointnet2x/checkpoint-ep200.pth.tar) | [config](./configs/point_within_format.yaml) |
| PointNet++-3x | 150 | 1024 | 64.1 | [model](https://dl.fbaipublicfiles.com/DepthContrast/pointnet3x/checkpoint-ep150.pth.tar) | [config](./configs/point_within_format.yaml) |
| PointNet++-4x | 100 | 1024 | 63.8 | [model](https://dl.fbaipublicfiles.com/DepthContrast/pointnet4x/checkpoint-ep100.pth.tar) | [config](./configs/point_within_format.yaml) |

The ScanNet detection evaluation metric is mAP at IOU=0.25. You need to change the scale parameter in the config files accordingly.

We provide the joint training results here, with different epochs. We use epoch 400 to generate the results reported in the paper.

| Backbone | epochs | batch-size | url | args |
|-------------------|-------------------|---------------------|--------------------|--------------------|
| PointNet++ & MinkowskiEngine UNet | 300 | 1024 | [model](https://dl.fbaipublicfiles.com/DepthContrast/pointnet_unet_joint/checkpoint-ep300.pth.tar) | [config](./configs/point_vox_template.yaml) |
| PointNet++ & MinkowskiEngine UNet | 400 | 1024 | [model](https://dl.fbaipublicfiles.com/DepthContrast/pointnet_unet_joint/checkpoint-ep400.pth.tar) | [config](./configs/point_vox_template.yaml) |
| PointNet++ & MinkowskiEngine UNet | 500 | 1024 | [model](https://dl.fbaipublicfiles.com/DepthContrast/pointnet_unet_joint/checkpoint-ep500.pth.tar) | [config](./configs/point_vox_template.yaml) |
| PointNet++ & MinkowskiEngine UNet | 600 | 1024 | [model](https://dl.fbaipublicfiles.com/DepthContrast/pointnet_unet_joint/checkpoint-ep600.pth.tar) | [config](./configs/point_vox_template.yaml) |
| PointNet++ & MinkowskiEngine UNet | 700 | 1024 | [model](https://dl.fbaipublicfiles.com/DepthContrast/pointnet_unet_joint/checkpoint-ep700.pth.tar) | [config](./configs/point_vox_template.yaml) |

# Running DepthContrast unsupervised training

## Requirements
You can use the requirements.txt to setup the environment.
First download the git-repo and install the pointnet modules:
```
git clone --recursive https://github.com/facebookresearch/DepthContrast.git 
cd pointnet2
python setup.py install
```
Then install all other packages:
```
pip install -r requirements.txt
```
or
```
conda install --file requirements.txt
```

For voxel representation, you have to install MinkowskiEngine. Please see [here](https://github.com/chrischoy/SpatioTemporalSegmentation) on how to install it.

For the lidar point cloud pretraining, we use models from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). It should be in the third_party folder. To install OpenPCDet, you need to install [spconv](https://github.com/traveller59/spconv), which is a bit difficult to install and may not be compatible with MinkowskiEngine. Thus, we suggest you use a different conda environment for lidar point cloud pretraining. 

## Singlenode training
DepthContrast is very simple to implement and experiment with.

To experiment with it on one GPU and debugging, you can do:
```
python main.py /path/to/cfg/file
```

For the actual training, please use the distributed trainer. 
For multi-gpu training in one node, you can run:
```
python main.py /path/to/cfg_file --multiprocessing-distributed --world-size 1 --rank 0 --ngpus number_of_gpus
```
To run it with just one gpu, just set the --ngpus to 1.
For submitting it to a slurm node, you can use ./scripts/pretrain_node1.sh. For hyper-parameter tuning, please change the config files.

## Multinode training
Distributed training is available via Slurm. We provide several [SBATCH scripts](./scripts) to reproduce our results.
For example, to train DepthContrast on 4 nodes and 32 GPUs with a batch size of 1024 run:
```
sbatch ./scripts/pretrain_node4.sh /path/to/cfg_file
```
Note that you might need to remove the copyright header from the sbatch file to launch it.

# Evaluating models
For votenet finetuning, please checkout this [repo](https://github.com/zaiweizhang/votenet) for more details.  

For H3DNet finetuning, please checkout this [repo](https://github.com/zaiweizhang/H3DNet) for more details.  

For voxel scene segmentation task finetuning, please checkout this [repo](https://github.com/zaiweizhang/SpatioTemporalSegmentation) for more details.  

For lidar point cloud object detection task finetuning, please checkout this [repo](https://github.com/zaiweizhang/OpenPCDet) for more details.  

# Common Issues
For help or issues using DepthContrast, please submit a GitHub issue.

## License
See the [LICENSE](LICENSE) file for more details.

## Citation
If you find this repository useful in your research, please cite:

```
@inproceedings{zhang_depth_contrast,
  title={Self-Supervised Pretraining of 3D Features on any Point-Cloud},
  author={Zhang, Zaiwei and Girdhar, Rohit and Joulin, Armand and Misra, Ishan},
  journal={arXiv preprint arXiv:2101.02691},
  year={2021}
}
```
