# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
import os
import tensorflow as tf
import numpy as np
import glob 
from open3d import *

tf.enable_eager_execution()

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


debug = False
K = 5
SCALE_FACTOR = 2
segments = glob.glob(sys.argv[1]+"/*")
datalist = []

def pc2obj(pc, filepath='test.obj'):
    pc = pc.T
    nverts = pc.shape[1]
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in range(nverts):
            f.write("v %.4f %.4f %.4f\n" % (pc[0,v],pc[1,v],pc[2,v]))

def extract(i):
  FILENAME = segments[i]
  run = FILENAME.split('segment-')[-1].split('.')[0]
  out_base_dir = sys.argv[2]+'/%s/' % run
 
  if not os.path.exists(out_base_dir):
    os.makedirs(out_base_dir)
    
  dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
  print(FILENAME)
  pc, pc_c = [], []
  camID2extr_v2c = {}
  camID2intr = {}
  
  all_static_pc = []
  for frame_cnt, data in enumerate(dataset):
    if frame_cnt % 2 != 0: continue ### Set the sampling rate here
    
    print('frame ', frame_cnt)
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))

    extr_laser2v = np.array(frame.context.laser_calibrations[0].extrinsic.transform).reshape(4, 4)
    extr_v2w = np.array(frame.pose.transform).reshape(4, 4)
    
    if frame_cnt == 0:
      for k in range(len(frame.context.camera_calibrations)):
        cameraID = frame.context.camera_calibrations[k].name
        extr_c2v =\
           np.array(frame.context.camera_calibrations[k].extrinsic.transform).reshape(4, 4)
        extr_v2c = np.linalg.inv(extr_c2v)
        camID2extr_v2c[frame.images[k].name] = extr_v2c
        fx = frame.context.camera_calibrations[k].intrinsic[0]
        fy = frame.context.camera_calibrations[k].intrinsic[1]
        cx = frame.context.camera_calibrations[k].intrinsic[2]
        cy = frame.context.camera_calibrations[k].intrinsic[3]
        k1 = frame.context.camera_calibrations[k].intrinsic[4]
        k2 = frame.context.camera_calibrations[k].intrinsic[5]
        p1 = frame.context.camera_calibrations[k].intrinsic[6]
        p2 = frame.context.camera_calibrations[k].intrinsic[7]
        k3 = frame.context.camera_calibrations[k].intrinsic[8]
        camID2intr[frame.images[k].name] = np.array([[fx, 0, cx],
                        [0, fy, cy], 
                        [0, 0, 1]])
        

    # lidar point cloud 
    
    (range_images, camera_projections, range_image_top_pose) = \
        frame_utils.parse_range_image_and_camera_projection(frame)
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                                                                      frame,
                                                                      range_images,
                                                                      camera_projections,
                                                                      range_image_top_pose)
                                                                      

    points_all = np.concatenate(points, axis=0)
    np.save('%s/frame_%03d.npy' % (out_base_dir, frame_cnt), points_all)
    datalist.append(os.path.abspath('%s/frame_%03d.npy' % (out_base_dir, frame_cnt)))
    
if __name__ == '__main__':
  for i in range(len(segments)):
      extract(i)
  np.save(sys.argv[3], datalist)
