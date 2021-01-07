#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from datasets.transforms import transforms

import numpy as np

collate_fn = transforms.cfl_collate_fn_factory(0)

def point_vox_moco_collator(batch):
    batch_size = len(batch)
    
    point = [x["data"] for x in batch]
    point_moco = [x["data_moco"] for x in batch]
    vox = [x["vox"] for x in batch]
    vox_moco = [x["vox_moco"] for x in batch]
    # labels are repeated N+1 times but they are the same
    labels = [x["label"][0] for x in batch]
    labels = torch.LongTensor(labels).squeeze()

    # data valid is repeated N+1 times but they are the same
    data_valid = torch.BoolTensor([x["data_valid"][0] for x in batch])

    points_moco = torch.stack([point_moco[i][0] for i in range(batch_size)])
    points = torch.stack([point[i][0] for i in range(batch_size)])
    
    vox_moco = collate_fn([vox_moco[i][0] for i in range(batch_size)])
    vox = collate_fn([vox[i][0] for i in range(batch_size)])

    output_batch = {
        "points": points,
        "points_moco": points_moco,
        "vox": vox,
        "vox_moco": vox_moco,
        "label": labels,
        "data_valid": data_valid,
    }

    return output_batch
