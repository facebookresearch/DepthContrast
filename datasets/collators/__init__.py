#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets.collators.point_moco_collator import point_moco_collator
try:
    from datasets.collators.vox_moco_collator import vox_moco_collator
    from datasets.collators.point_vox_moco_collator import point_vox_moco_collator
except:
    print ("Cannot import minkowski engine. Try spconv next")
    from datasets.collators.point_vox_moco_lidar_collator import point_vox_moco_collator
from torch.utils.data.dataloader import default_collate


COLLATORS_MAP = {
    "default": default_collate,
    "point_moco_collator": point_moco_collator,
    "point_vox_moco_collator": point_vox_moco_collator,
}


def get_collator(name):
    assert name in COLLATORS_MAP, "Unknown collator"
    return COLLATORS_MAP[name]


__all__ = ["get_collator"]
