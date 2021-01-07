# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

from models.trunks.mlp import MLP

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ROOT_DIR)
ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'third_party', 'OpenPCDet', "pcdet"))

from ops.pointnet2.pointnet2_batch import pointnet2_modules

class PointNet2MSG(nn.Module):
    def __init__(self, use_mlp=False, mlp_dim=None):
        super().__init__()

        input_channels = 4
        
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        SA_CONFIG = {'NPOINTS': [4096, 1024, 256, 64], 'RADIUS': [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]], 'NSAMPLE': [[16, 32], [16, 32], [16, 32], [16, 32]], 'MLPS': [[[16, 16, 32], [32, 32, 64]], [[64, 64, 128], [64, 96, 128]], [[128, 196, 256], [128, 196, 256]], [[256, 256, 512], [256, 384, 512]]]}

        FP_MLPS = [[128, 128], [256, 256], [512, 512], [512, 512]]
        
        for k in range(SA_CONFIG["NPOINTS"].__len__()):
            mlps = SA_CONFIG["MLPS"][k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG(
                    npoint=SA_CONFIG["NPOINTS"][k],
                    radii=SA_CONFIG["RADIUS"][k],
                    nsamples=SA_CONFIG["NSAMPLE"][k],
                    mlps=mlps,
                    use_xyz=True,
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules.PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k]
                )
            )

        self.num_point_features = FP_MLPS[0][-1]
        
        self.all_feat_names = [
            "fp2",
        ]

        if use_mlp:
            self.use_mlp = True
            self.head = MLP(mlp_dim)

            
    def break_up_pc(self, pc):
        #batch_idx = pc[:, 0]
        xyz = pc[:, :, 0:3].contiguous()
        features = (pc[:, :, 3:].contiguous() if pc.size(-1) > 3 else None)
        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, out_feat_keys=None):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = pointcloud.shape[0]
        points = pointcloud
        xyz, features = self.break_up_pc(points)

        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            assert l_features[i].is_contiguous()
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            assert li_features.is_contiguous()
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            assert l_features[i - 1].is_contiguous()
            assert l_features[i].is_contiguous()
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )  # (B, C, N)
            assert l_features[i - 1].is_contiguous()

        point_features = l_features[0]
        
        end_points = {}
        end_points['fp2_features'] = point_features
    
        out_feats = [None] * len(out_feat_keys)
        for key in out_feat_keys:
            feat = end_points[key+"_features"]
            nump = feat.shape[-1]
            
            feat = torch.squeeze(F.max_pool1d(feat, nump))
            if self.use_mlp:
                feat = self.head(feat)
            out_feats[out_feat_keys.index(key)] = feat
        
        return out_feats
