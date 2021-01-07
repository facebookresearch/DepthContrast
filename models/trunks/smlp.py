#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import MinkowskiEngine as ME

class SMLP(nn.Module):
    def __init__(
        self, dims, use_bn=False, use_relu=False, use_dropout=False, use_bias=True
    ):
        super().__init__()
        layers = []
        last_dim = dims[0]
        counter = 1
        for dim in dims[1:]:
            layers.append(ME.MinkowskiLinear(last_dim, dim, bias=use_bias))
            counter += 1
            if use_bn:
                layers.append(
                    ME.MinkowskiBatchNorm(
                        dim,
                        eps=1e-5,
                        momentum=0.1,
                    )
                )
            if (counter < len(dims)) and use_relu:
                layers.append(ME.MinkowskiReLU(inplace=True))
                last_dim = dim
            if use_dropout:
                layers.append(MinkowskiDropout.Dropout())
        self.clf = nn.Sequential(*layers)

    def forward(self, batch):
        out = self.clf(batch)
        return out
