# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import datetime
import sys

import torch
from torch import distributed as dist


class Logger(object):
    def __init__(self, quiet=False, log_fn=None, rank=0, prefix=""):
        self.rank = rank if rank is not None else 0
        self.quiet = quiet
        self.log_fn = log_fn

        self.prefix = ""
        if prefix:
            self.prefix = prefix + ' | '

        self.file_pointers = []
        if self.rank == 0:
            if self.quiet:
                open(log_fn, 'w').close()

    def add_line(self, content):
        if self.rank == 0:
            msg = self.prefix+content
            if self.quiet:
                fp = open(self.log_fn, 'a')
                fp.write(msg+'\n')
                fp.flush()
                fp.close()
            else:
                print(msg)
                sys.stdout.flush()


class ProgressMeter(object):
    def __init__(self, num_batches, meters, phase, epoch=None, logger=None, tb_writter=None):
        self.batches_per_epoch = num_batches
        self.batch_fmtstr = self._get_batch_fmtstr(epoch, num_batches)
        self.meters = meters
        self.phase = phase
        self.epoch = epoch
        self.logger = logger
        self.tb_writter = tb_writter

    def display(self, batch):
        step = self.epoch * self.batches_per_epoch + batch
        date = str(datetime.datetime.now())
        entries = ['{} | {} {}'.format(date, self.phase, self.batch_fmtstr.format(batch))]
        entries += [str(meter) for meter in self.meters]
        if self.logger is None:
            print('\t'.join(entries))
        else:
            self.logger.add_line('\t'.join(entries))

        if self.tb_writter is not None:
            for meter in self.meters:
                self.tb_writter.add_scalar('{}-batch/{}'.format(self.phase, meter.name), meter.val, step)

    def _get_batch_fmtstr(self, epoch, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        epoch_str = '[{}]'.format(epoch) if epoch is not None else ''
        return epoch_str+'[' + fmt + '/' + fmt.format(num_batches) + ']'

    def synchronize_meters(self, cur_gpu):
        metrics = torch.tensor([m.avg for m in self.meters]).cuda(cur_gpu)
        metrics_gather = [torch.ones_like(metrics) for _ in range(dist.get_world_size())]
        dist.all_gather(metrics_gather, metrics)

        metrics = torch.stack(metrics_gather).mean(0).cpu().numpy()
        for meter, m in zip(self.meters, metrics):
            meter.avg = m
