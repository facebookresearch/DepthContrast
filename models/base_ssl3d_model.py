#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from SparseTensor import SparseTensor
    import MinkowskiEngine as ME
except:
    pass

import numpy as np

from utils import main_utils

def parameter_description(model):
    desc = ''
    for n, p in model.named_parameters():
        desc += "{:70} | {:10} | {:30} | {}\n".format(
            n, 'Trainable' if p.requires_grad else 'Frozen',
            ' x '.join([str(s) for s in p.size()]), str(np.prod(p.size())))
    return desc

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class BaseSSLMultiInputOutputModel(nn.Module):
    def __init__(self, model_config, logger):
        """
        Class to implement a self-supervised model.
        The model is split into `trunk' that computes features.
        """
        self.config = model_config
        self.logger = logger
        super().__init__()
        self.eval_mode = None  # this is just informational
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.trunk = self._get_trunk()
        self.m = 0.999 ### Can be tuned
        self.model_input = model_config["model_input"]
        self.model_feature = model_config["model_feature"]
        
    def multi_input_with_head_mapping_forward(self, batch):
        all_outputs = []
        for input_idx in range(len(self.model_input)):
            input_key = self.model_input[input_idx]
            feature_names = self.model_feature[input_idx]
            if "moco" in input_key:
                outputs = self._single_input_forward_MOCO(batch[input_key], feature_names, input_key, input_idx)
            else:
                outputs = self._single_input_forward(batch[input_key], feature_names, input_key, input_idx)
            if len(outputs) == 1:
                # single head. do not make nested list
                outputs = outputs[0]
            else:
                all_outputs += outputs
                continue
            all_outputs.append(outputs)
        return all_outputs
    
    def _single_input_forward(self, batch, feature_names, input_key, target):
        if "vox" not in input_key:
            assert isinstance(batch, torch.Tensor)

        if ('vox' in input_key) and ("Lidar" not in self.config):
            points = batch
            points_coords = points[0]
            points_feats = points[1]

            ### Invariant to even and odd coords
            points_coords[:, 1:] += (torch.rand(3) * 100).type_as(points_coords)
            points_feats = points_feats/255.0 - 0.5

            batch = SparseTensor(points_feats, points_coords.float())

        if ('vox' in input_key) and ("Lidar" in self.config):
            # Copy to GPU
            for key in batch:
                batch[key] = main_utils.recursive_copy_to_gpu(
                    batch[key], non_blocking=True
                )
        else:
            # Copy to GPU
            batch = main_utils.recursive_copy_to_gpu(
                batch, non_blocking=True
            )
        
        feats = self.trunk[target](batch, feature_names)
        return feats

    @torch.no_grad()
    def _momentum_update_key(self, target=1):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.trunk[target-1].parameters(), self.trunk[target].parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, vox=False, idx_shuffle=None):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        if vox:
            batch_size = []
            for bidx in x:
                batch_size.append(len(bidx))
            all_size = concat_all_gather(torch.tensor(batch_size).cuda())
            max_size = torch.max(all_size)

            ### Change the new size here
            newx = []
            for bidx in range(len(x)):
                newx.append(torch.ones((max_size, x[bidx].shape[1])).cuda())
                newx[bidx][:len(x[bidx]),:] = x[bidx]
            newx = torch.stack(newx)
            batch_size_this = newx.shape[0]
        else:
            batch_size_this = x.shape[0]

        if vox:
            x_gather = concat_all_gather(newx)
        else:
            x_gather = concat_all_gather(x)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        if idx_shuffle == None:
            # random shuffle index
            idx_shuffle = torch.randperm(batch_size_all).cuda()
        
            # broadcast to all gpus
            torch.distributed.broadcast(idx_shuffle, src=0)
        
        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)
        
        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        if vox:
            ret_x = []
            batch_idx = []
            for idx in range(len(idx_this)):
                if x_gather.shape[-1] == 4:
                    ### Change the batch index here
                    tempdata = x_gather[idx_this[idx]][:all_size[idx_this[idx]],:]
                    tempdata[:,0] = idx
                    ret_x.append(tempdata)
                else:
                    ret_x.append(x_gather[idx_this[idx]][:all_size[idx_this[idx]],:])
            ret_x = torch.cat(ret_x)
            return ret_x, idx_unshuffle, idx_shuffle
        else:
            return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        
        num_gpus = batch_size_all // batch_size_this
        
        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this]
    
    def _single_input_forward_MOCO(self, batch, feature_names, input_key, target):
        if "vox" not in input_key:
            assert isinstance(batch, torch.Tensor)
        if ('vox' in input_key) and ("Lidar" not in self.config):
            points = batch
            points_coords = points[0]
            points_feats = points[1]

            ### Invariant to even and odd coords
            points_coords[:, 1:] += (torch.rand(3) * 100).type_as(points_coords)
            points_feats = points_feats/255.0 - 0.5
            ### If enable shuffle batch for vox, please comment out this line.
            batch = SparseTensor(points_feats, points_coords.float())

        with torch.no_grad():
            self._momentum_update_key(target)  # update the key encoder
            # shuffle for making use of BN
            if torch.distributed.is_initialized():
                if "vox" not in input_key: 
                    batch, idx_unshuffle = self._batch_shuffle_ddp(batch, vox=False)
                if False:
                    ### Skip batch shuffle for vox for now
                    ### Does not give performance gain
                    if ("Lidar" not in self.config):
                        batch_inds = points_coords[:,0].detach().cpu().numpy()
                        points_coords = main_utils.recursive_copy_to_gpu(
                            points_coords, non_blocking=True
                        )
                        points_feats = main_utils.recursive_copy_to_gpu(
                            points_feats, non_blocking=True
                        )
                        
                        point_coord_split = []
                        point_feat_split = []
                    
                        for batch_ind in np.unique(batch_inds):
                            point_coord_split.append(points_coords[points_coords[:,0]==batch_ind])
                            point_feat_split.append(points_feats[points_coords[:,0]==batch_ind])
                    
                        points_coords, idx_unshuffle, idx_shuffle = self._batch_shuffle_ddp(point_coord_split, vox=True)
                        points_feats, _, _ = self._batch_shuffle_ddp(point_feat_split, vox=True, idx_shuffle=idx_shuffle)
                        batch = SparseTensor(points_feats, points_coords.float())
                    else:
                        print ("Not implemented yet")    
            else:
                if ('vox' in input_key) and ("Lidar" not in self.config):
                    batch = SparseTensor(points_feats, points_coords.float())
                
            # Copy to GPU
            if ("Lidar" in self.config) and ("vox" in input_key):
                for key in batch:
                    batch[key] = main_utils.recursive_copy_to_gpu(
                        batch[key], non_blocking=True
                    )
            else:
                batch = main_utils.recursive_copy_to_gpu(
                    batch, non_blocking=True
                )
            
            feats = self.trunk[target](batch, feature_names)
            if torch.distributed.is_initialized():
                if "vox" not in input_key:
                    feats = [self._batch_unshuffle_ddp(feats[0], idx_unshuffle)]
                return feats
            else:
                return feats

    def forward(self, batch):
        return self.multi_input_with_head_mapping_forward(batch)

    def _get_trunk(self):
        import models.trunks as models
        trunks = torch.nn.ModuleList()
        if 'arch_point' in self.config:
            assert self.config['arch_point'] in models.TRUNKS, 'Unknown model architecture'
            trunks.append(models.TRUNKS[self.config['arch_point']](**self.config['args_point']))
            trunks.append(models.TRUNKS[self.config['arch_point']](**self.config['args_point']))
        if 'arch_vox' in self.config:
            assert self.config['arch_vox'] in models.TRUNKS, 'Unknown model architecture'
            trunks.append(models.TRUNKS[self.config['arch_vox']](**self.config['args_vox']))
            trunks.append(models.TRUNKS[self.config['arch_vox']](**self.config['args_vox']))

        for numh in range(len(trunks)//2):
            for param_q, param_k in zip(trunks[numh*2].parameters(), trunks[numh*2+1].parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

        logger = self.logger
        for model in trunks:
            if logger is not None:
                if isinstance(model, (list, tuple)):
                    logger.add_line("=" * 30 + "   Model   " + "=" * 30)
                    for m in model:
                        logger.add_line(str(m))
                    logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
                    for m in model:
                        logger.add_line(parameter_description(m))
                else:
                    logger.add_line("=" * 30 + "   Model   " + "=" * 30)
                    logger.add_line(str(model))
                    logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
                    logger.add_line(parameter_description(model))
        return trunks

    def _print_state_dict_shapes(self, state_dict):
        logging.info("Model state_dict:")
        for param_tensor in state_dict.keys():
            logging.info(f"{param_tensor}:\t{state_dict[param_tensor].size()}")

    def _print_loaded_dict_info(self, state):
        # get the model state dict original
        model_state_dict = {}
        if "," in self.config.TRUNK.NAME:
            trunk_state_dict, heads_state_dict = (
                self.trunk.state_dict(),
                self.heads.state_dict(),
            )
        else:
            trunk_state_dict, heads_state_dict = (
                self.trunk.state_dict(),
                self.heads.state_dict(),
            )
        model_state_dict.update(trunk_state_dict)
        model_state_dict.update(heads_state_dict)

        # get the checkpoint state dict
        checkpoint_state_dict = {}
        checkpoint_state_dict.update(state["trunk"])
        checkpoint_state_dict.update(state["heads"])

        # now we compare the state dict and print information
        not_found, extra_layers = [], []
        max_len_model = max(len(key) for key in model_state_dict.keys())
        for layername in model_state_dict.keys():
            if layername in checkpoint_state_dict:
                logging.info(
                    f"Loaded: {layername: <{max_len_model}} of "
                    f"shape: {model_state_dict[layername].size()} from checkpoint"
                )
            else:
                not_found.append(layername)
                logging.info(f"Not found:\t\t{layername}, not initialized")
        for layername in checkpoint_state_dict.keys():
            if layername not in model_state_dict:
                extra_layers.append(layername)
        logging.info(f"Extra layers not loaded from checkpoint:\n {extra_layers}")

    def get_optimizer_params(self):
        regularized_params, unregularized_params = [], []
        conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
        bn_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            apex.parallel.SyncBatchNorm,
        )
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, conv_types):
                regularized_params.append(module.weight)
                if module.bias is not None:
                    if self.optimizer_config["regularize_bias"]:
                        regularized_params.append(module.bias)
                    else:
                        unregularized_params.append(module.bias)
            elif isinstance(module, bn_types):
                if module.weight is not None:
                    if self.optimizer_config["regularize_bn"]:
                        regularized_params.append(module.weight)
                    else:
                        unregularized_params.append(module.weight)
                if module.bias is not None:
                    if (
                        self.optimizer_config["regularize_bn"]
                        and self.optimizer_config["regularize_bias"]
                    ):
                        regularized_params.append(module.bias)
                    else:
                        unregularized_params.append(module.bias)
            elif len(list(module.children())) >= 0:
                # for any other layers not bn_types, conv_types or nn.Linear, if
                # the layers are the leaf nodes and have parameters, we regularize
                # them. Similarly, if non-leaf nodes but have parameters, regularize
                # them (set recurse=False)
                for params in module.parameters(recurse=False):
                    regularized_params.append(params)

        non_trainable_params = []
        for name, param in self.named_parameters():
            if name in cfg.MODEL.NON_TRAINABLE_PARAMS:
                param.requires_grad = False
                non_trainable_params.append(param)

        trainable_params = [
            params for params in self.parameters() if params.requires_grad
        ]
        regularized_params = [
            params for params in regularized_params if params.requires_grad
        ]
        unregularized_params = [
            params for params in unregularized_params if params.requires_grad
        ]
        logging.info("Traininable params: {}".format(len(trainable_params)))
        logging.info("Non-Traininable params: {}".format(len(non_trainable_params)))
        logging.info(
            "Regularized Parameters: {}. Unregularized Parameters {}".format(
                len(regularized_params), len(unregularized_params)
            )
        )
        return {
            "regularized_params": regularized_params,
            "unregularized_params": unregularized_params,
        }

    @property
    def num_classes(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    @property
    def model_depth(self):
        raise NotImplementedError

    def validate(self, dataset_output_shape):
        raise NotImplementedError
