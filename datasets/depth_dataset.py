# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import os

import numpy as np

from datasets.transforms.augment3d import get_transform3d

try:
    ### Default uses minkowski engine
    from datasets.transforms.voxelizer import Voxelizer
    from datasets.transforms import transforms
except:
    pass
    
try:
    try:
        from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
    except:
        from spconv.utils import VoxelGenerator
except:
    pass

from torch.utils.data import Dataset
import torch

### Waymo lidar range
#POINT_RANGE = np.array([  0. , -75. ,  -3. ,  75.0,  75. ,   3. ], dtype=np.float32)
POINT_RANGE = np.array([0, -40, -3, 70.4, 40, 1], dtype=np.float32)#np.array([  0. , -75. ,  -3. ,  75.0,  75. ,   3. ], dtype=np.float32) ### KITTI

class DepthContrastDataset(Dataset):
    """Base Self Supervised Learning Dataset Class."""

    def __init__(self, cfg):
        self.split = "train" ### Default is training
        self.label_objs = []
        self.data_paths = []
        self.label_paths = []
        self.cfg = cfg
        self.batchsize_per_replica = cfg["BATCHSIZE_PER_REPLICA"]
        self.label_sources = []#cfg["LABEL_SOURCES"]
        self.dataset_names = cfg["DATASET_NAMES"]
        self.label_type = cfg["LABEL_TYPE"]
        self.AUGMENT_COORDS_TO_FEATS = False #optional
        self._labels_init = False
        self._get_data_files("train")
        self.data_objs = np.load(self.data_paths[0]) ### Only load the first one for now

        #### Add the voxelizer here
        if ("Lidar" in cfg) and cfg["VOX"]:
            self.VOXEL_SIZE = [0.1, 0.1, 0.2]
       
            self.point_cloud_range = POINT_RANGE#np.array([  0. , -75. ,  -3. ,  75.0,  75. ,   3. ], dtype=np.float32)
            self.MAX_POINTS_PER_VOXEL = 5
            self.MAX_NUMBER_OF_VOXELS = 16000
            self.voxel_generator = VoxelGenerator(
                voxel_size=self.VOXEL_SIZE,
                point_cloud_range=self.point_cloud_range,
                max_num_points=self.MAX_POINTS_PER_VOXEL,
                max_voxels=self.MAX_NUMBER_OF_VOXELS
            )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = self.VOXEL_SIZE
        elif cfg["VOX"]:
            augment_data = (self.split == "TRAIN")
            #### Vox parameters here
            self.VOXEL_SIZE = 0.05 #0.02 # 5cm
            self.CLIP_BOUND = None#(-1000, -1000, -1000, 1000, 1000, 1000)

            self.data_aug_color_trans_ratio = 0.1
            self.data_aug_color_jitter_std = 0.05
            self.ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))
            
            if augment_data:
                self.prevoxel_transform_train = []
                self.prevoxel_transform_train.append(transforms.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS))
                self.prevoxel_transform = transforms.Compose(self.prevoxel_transform_train)
                
                self.input_transforms = []
                self.input_transforms += [
                    transforms.RandomDropout(0.2),
                    transforms.RandomHorizontalFlip('z', False),
                    #transforms.ChromaticAutoContrast(),
                    transforms.ChromaticTranslation(self.data_aug_color_trans_ratio),
                    transforms.ChromaticJitter(self.data_aug_color_jitter_std),
                    # t.HueSaturationTranslation(config.data_aug_hue_max, config.data_aug_saturation_max),
                ]
                self.input_transforms = transforms.Compose(self.input_transforms)
                
            # Coordinate Augmentation Arguments: Unlike feature augmentation, coordinate
            # augmentation has to be done before voxelization
            self.SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
            self.ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,np.pi))
            self.TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
            
            self.voxelizer = Voxelizer(
                voxel_size=self.VOXEL_SIZE,
                clip_bound=self.CLIP_BOUND,
                use_augmentation=augment_data,
                scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
                rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
                translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND,
                ignore_label=True)

    def _get_data_files(self, split):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        self.data_paths = self.cfg["DATA_PATHS"]
        self.label_paths = []
        
        logging.info(f"Rank: {local_rank} Data files:\n{self.data_paths}")
        logging.info(f"Rank: {local_rank} Label files:\n{self.label_paths}")

    def _augment_coords_to_feats(self, coords, feats, labels=None):
        # Center x,y
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = coords - coords_center
        feats = np.concatenate((feats, norm_coords), 1)
        return coords, feats, labels

    def toVox(self, coords, feats, labels):
        if "Lidar" in self.cfg:
            voxel_output = self.voxel_generator.generate(coords)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                                                  voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output

            data_dict = {}
            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points
            return data_dict
        else:
            precoords = np.copy(coords)
            prefeats = np.copy(feats)
            if (self.split == "TRAIN") and (self.prevoxel_transform is not None):
                coords, feats, labels = self.prevoxel_transform(coords, feats, labels)
            coords, feats, labels, transformation = self.voxelizer.voxelize(coords, feats, labels)
            if (self.split == "TRAIN") and (self.input_transforms is not None):
                try:
                    coords, feats, labels = self.input_transforms(coords, feats, labels)
                except:
                    print ("error with: ", coords.shape)
                    coords = np.zeros((100,3),dtype=np.int32)
                    feats = np.zeros((100,3),dtype=np.float64)
                    labels = np.zeros((100,),dtype=np.int32)
            if (self.split == "TRAIN") and (self.AUGMENT_COORDS_TO_FEATS):
                coords, feats, labels = self._augment_coords_to_feats(coords, feats, labels)
            return (coords, feats, labels)

    def load_data(self, idx):
        is_success = True
        point_path = self.data_objs[idx]
        try:
            if "Lidar" in self.cfg:
                #point = np.load(point_path)
                point = np.fromfile(str(point_path), dtype=np.float32).reshape(-1, 4)#np.load(point_path)
                if point.shape[1] != 4:
                    temp = np.zeros((point.shape[0],4))
                    temp[:,:3] = point
                    point = np.copy(temp)

                upper_idx = np.sum((point[:,0:3] <= POINT_RANGE[3:6]).astype(np.int32), 1) == 3
                lower_idx = np.sum((point[:,0:3] >= POINT_RANGE[0:3]).astype(np.int32), 1) == 3

                new_pointidx = (upper_idx) & (lower_idx)
                point = point[new_pointidx,:]
            else:
                point = np.load(point_path)
                ### Add height
                floor_height = np.percentile(point[:,2],0.99)
                height = point[:,2] - floor_height
                point = np.concatenate([point, np.expand_dims(height, 1)],1)
        except Exception as e:
            logging.warn(
                f"Couldn't load: {self.point_dataset[idx]}. Exception: \n{e}"
            )
            point = np.zeros([50000, 7])
            is_success = False
        return point, is_success

    def __getitem__(self, idx):

        cfg = self.cfg
        # TODO: this doesn't yet handle the case where the length of datasets
        # could be different.
        if cfg["DATA_TYPE"] == "point_vox":
            item = {"data": [], "data_valid": [], "data_moco": [], "vox": [], "vox_moco": []}

            data, valid = self.load_data(idx)
            item["data"].append(data)
            item["data_moco"].append(np.copy(data))
            item["vox"].append(np.copy(data))
            item["vox_moco"].append(np.copy(data))
            item["data_valid"].append(1 if valid else -1)
        else:
            item = {"data": [], "data_moco": [], "data_valid": [], "data_idx": []}
            
            data, valid = self.load_data(idx)
            item["data"].append(data)
            item["data_moco"].append(np.copy(data))
            item["data_valid"].append(1 if valid else -1)

        ### Make copies for moco setting
        item["label"] = []
        item["label"].append(idx)

        ### Apply the transformation here
        if (cfg["DATA_TYPE"] == "point_vox"):
            tempitem = {"data": item["data"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"])
            item["data"] = tempdata["data"]

            tempitem = {"data": item["data_moco"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"])
            item["data_moco"] = tempdata["data"]

            tempitem = {"data": item["vox"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=True)
            coords = tempdata["data"][0][:,:3]
            feats = tempdata["data"][0][:,3:6]*255.0#np.ones(coords.shape)*255.0
            labels = np.zeros(coords.shape[0]).astype(np.int32)
            item["vox"] = [self.toVox(coords, feats, labels)]
            
            tempitem = {"data": item["vox_moco"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=True)
            coords = tempdata["data"][0][:,:3]
            feats = tempdata["data"][0][:,3:6]*255.0#np.ones(coords.shape)*255.0
            labels = np.zeros(coords.shape[0]).astype(np.int32)                    
            item["vox_moco"] = [self.toVox(coords, feats, labels)]               
        else:
            tempitem = {"data": item["data"]}
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=cfg["VOX"])
            if cfg["VOX"]:
                coords = tempdata["data"][0][:,:3]
                feats = tempdata["data"][0][:,3:6]*255.0
                labels = np.zeros(coords.shape[0]).astype(np.int32)
                item["data"] = [self.toVox(coords, feats, labels)]
            else:
                item["data"] = tempdata["data"]
    
            tempitem = {"data": item["data_moco"]}                
            tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=cfg["VOX"])
            if cfg["VOX"]:
                coords = tempdata["data"][0][:,:3]
                feats = tempdata["data"][0][:,3:6]*255.0#np.ones(coords.shape)*255.0
                labels = np.zeros(coords.shape[0]).astype(np.int32)                    
                item["data_moco"] = [self.toVox(coords, feats, labels)]
            else:
                item["data_moco"] = tempdata["data"]

        return item

    def __len__(self):
        return len(self.data_objs)

    def get_available_splits(self, dataset_config):
        return [key for key in dataset_config if key.lower() in ["train", "test"]]

    def num_samples(self, source_idx=0):
        return len(self.data_objs)

    def get_batchsize_per_replica(self):
        # this searches for batchsize_per_replica in self and then in self.dataset
        return getattr(self, "batchsize_per_replica", 1)

    def get_global_batchsize(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        return self.get_batchsize_per_replica() * world_size
