#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
try:
    from models.trunks.pointnet import PointNet
except:
    PointNet = None

try:
    from models.trunks.spconv.models.res16unet import Res16UNet34
except:
    Res16UNet34 = None
    
try:
    from models.trunks.pointnet2_backbone import PointNet2MSG
    from models.trunks.spconv_unet import UNetV2_concat as UNetV2
except:
    PointNet2MSG = None
    UNetV2 = None
    
TRUNKS = {
    "pointnet": PointNet,
    "unet": Res16UNet34,
    "pointnet_msg": PointNet2MSG,
    "UNetV2": UNetV2,
}


__all__ = ["TRUNKS"]
