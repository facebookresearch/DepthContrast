#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging

import torch
from datasets.collators import get_collator
from datasets.depth_dataset import DepthContrastDataset
from torch.utils.data import DataLoader


__all__ = ["DepthContrastDataset", "get_data_files"]


def build_dataset(cfg):
    dataset = DepthContrastDataset(cfg)
    return dataset


def print_sampler_config(data_sampler):
    sampler_cfg = {
        "num_replicas": data_sampler.num_replicas,
        "rank": data_sampler.rank,
        "epoch": data_sampler.epoch,
        "num_samples": data_sampler.num_samples,
        "total_size": data_sampler.total_size,
        "shuffle": data_sampler.shuffle,
    }
    if hasattr(data_sampler, "start_iter"):
        sampler_cfg["start_iter"] = data_sampler.start_iter
    if hasattr(data_sampler, "batch_size"):
        sampler_cfg["batch_size"] = data_sampler.batch_size
    logging.info("Distributed Sampler config:\n{}".format(sampler_cfg))


def get_loader(dataset, dataset_config, num_dataloader_workers, pin_memory):
    data_sampler = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        assert torch.distributed.is_initialized(), "Torch distributed isn't initalized"
        data_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        logging.info("Created the Distributed Sampler....")
        print_sampler_config(data_sampler)
    else:
        logging.warning(
            "Distributed trainer not initialized. Not using the sampler and data will NOT be shuffled"  # NOQA
        )
    collate_function = get_collator(dataset_config["COLLATE_FUNCTION"])
    dataloader = DataLoader(
        dataset=dataset,
        num_workers=num_dataloader_workers,
        pin_memory=pin_memory,
        shuffle=False,
        batch_size=dataset_config["BATCHSIZE_PER_REPLICA"],
        collate_fn=collate_function,
        sampler=data_sampler,
        drop_last=dataset_config["DROP_LAST"],
    )
    return dataloader
