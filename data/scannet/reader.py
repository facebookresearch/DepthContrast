# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os, sys

from SensorData import SensorData

from glob import glob

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--scans_path', required=True, help='path to scans folder')
parser.add_argument('--output_path', required=True, help='path to output folder')
parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
parser.add_argument('--export_poses', dest='export_poses', action='store_true')
parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
parser.set_defaults(export_depth_images=False, export_color_images=False, export_poses=False, export_intrinsics=False)

opt = parser.parse_args()
print(opt)


def main():
  scans = glob(opt.scans_path+"/*")
  for scan in scans:
    scenename = scan.split("/")[-1]
    filename = os.path.join(scan, scenename+".sens")
    if not os.path.exists(opt.output_path):
      os.makedirs(opt.output_path)
      os.makedirs(os.path.join(opt.output_path, 'depth'))
      os.makedirs(os.path.join(opt.output_path, 'color'))
      os.makedirs(os.path.join(opt.output_path, 'pose'))
      os.makedirs(os.path.join(opt.output_path, 'intrinsic'))
      # load the data
    sys.stdout.write('loading %s...' % filename)
    sd = SensorData(filename)
    sys.stdout.write('loaded!\n')
    if opt.export_depth_images:
      sd.export_depth_images(os.path.join(opt.output_path, 'depth', scenename))
    if opt.export_color_images:
      sd.export_color_images(os.path.join(opt.output_path, 'color', scenename))
    if opt.export_poses:
      sd.export_poses(os.path.join(opt.output_path, 'pose', scenename))
    if opt.export_intrinsics:
      sd.export_intrinsics(os.path.join(opt.output_path, 'intrinsic', scenename))


if __name__ == '__main__':
    main()
