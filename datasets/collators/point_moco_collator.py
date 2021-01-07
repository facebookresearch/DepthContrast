#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

def point_moco_collator(batch):
    batch_size = len(batch)
    
    data_point = [x["data"] for x in batch]
    data_moco = [x["data_moco"] for x in batch]
    # labels are repeated N+1 times but they are the same
    labels = [x["label"][0] for x in batch]
    labels = torch.LongTensor(labels).squeeze()
    
    # data valid is repeated N+1 times but they are the same
    data_valid = torch.BoolTensor([x["data_valid"][0] for x in batch])

    points_moco = torch.stack([data_moco[i][0] for i in range(batch_size)])
    points = torch.stack([data_point[i][0] for i in range(batch_size)])
    
    output_batch = {
        "points": points,
        "points_moco": points_moco,
        "label": labels,
        "data_valid": data_valid,
    }
    
    return output_batch
