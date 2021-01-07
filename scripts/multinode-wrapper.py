# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys, os

num_nodes = int(os.environ['SLURM_NNODES'])
node_id = int(os.environ['SLURM_NODEID'])
node0 = 'learnfair' + os.environ['SLURM_NODELIST'][10:14]
cmd = 'python {script} {cfg} --dist-url tcp://{node0}:1234 --dist-backend nccl --multiprocessing-distributed --world-size {ws} --rank {rank}'.format(
       script=sys.argv[1], cfg=sys.argv[2], node0=node0, ws=num_nodes, rank=node_id)
os.system(cmd)
