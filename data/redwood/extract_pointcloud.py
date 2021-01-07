# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import open3d as o3d
import cv2
import sys
import numpy as np
from glob import glob
import os
import zipfile

def pc2obj(pc, filepath='test.obj'):
    pc = pc.T
    nverts = pc.shape[1]
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in range(nverts):
            f.write("v %.4f %.4f %.4f\n" % (pc[0,v],pc[1,v],pc[2,v]))

nump = 50000 ### Number of points in the point clouds
scenelist = glob(sys.argv[1]+"*.zip") ### Path to the zip files
datalist = []

for scene in scenelist:
    if os.path.exists(sys.argv[3]+scene.split("/")[-1].split(".")[0]):
        continue

    os.system("rm -rf test/")
    with zipfile.ZipFile(scene, 'r') as zip_ref:
        zip_ref.extractall("test")
    
    os.system("ls test/rgb/ > rgblist")
    os.system("ls test/depth/ > depthlist")

    rgb_path = {}
    depth_path = {}
    rgblist = open("rgblist", "r")
    depthlist = open("depthlist", "r")
    for line in rgblist:
        seqname = line.split("-")[0]
        if seqname in rgb_path:
            print (line)
        else:
            rgb_path[seqname] = line.strip("\n")

    for line in depthlist:
        seqname = line.split("-")[0]
        if seqname in depth_path:
            print (line)
        else:
            depth_path[seqname] = line.strip("\n")
    
    intrin_cam = o3d.camera.PinholeCameraIntrinsic()
    #print (scene)
    if not os.path.exists(sys.argv[2]+scene.split("/")[-1].split(".")[0]):
        os.mkdir(sys.argv[2]+scene.split("/")[-1].split(".")[0])
    if not os.path.exists(sys.argv[3]+scene.split("/")[-1].split(".")[0]):
        os.mkdir(sys.argv[3]+scene.split("/")[-1].split(".")[0])
        
    success = 0

    if len(rgb_path) <= len(depth_path):
        framelist = rgb_path.keys()
    else:
        framelist = depth_path.keys()
    counter = 15
    for frame in framelist:
        counter += 1
        if counter < 15: ### Set the sampling rate here
            continue
        depth = "test/depth/"+depth_path[frame]

        rgbpath = "test/rgb/"+rgb_path[frame]#scene+"/matterport_color_images/"+room_name+"_i%d_%d.jpg" % (cam_num, frame_num)

        depth_im = cv2.imread(depth, -1)

        try:
            o3d_depth = o3d.geometry.Image(depth_im)
            rgb_im = cv2.imread(rgbpath)
            o3d_rgb = o3d.geometry.Image(rgb_im)
            o3d_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb, o3d_depth, depth_scale=1000.0, depth_trunc=1000.0, convert_rgb_to_intensity=False)
        except:
            print (frame)
            continue

        if (depth_im.shape[1] != 640) or (depth_im.shape[0] != 480):
            print (frame)
            continue
        intrin_cam.set_intrinsics(width=depth_im.shape[1], height=depth_im.shape[0], fx=525.0, fy=525.0, cx=319.5, cy=239.5)

        pts = o3d.geometry.PointCloud.create_from_rgbd_image(o3d_rgbd, intrin_cam, np.eye(4))

        if len(np.array(pts.points)) < 100:
            continue
        
        if len(np.array(pts.points)) >= nump:
            sel_idx = np.random.choice(len(np.array(pts.points)), nump, replace=False)
        else:
            sel_idx = np.random.choice(len(np.array(pts.points)), nump, replace=True)
        temp = np.array(pts.points)[sel_idx]

        if np.isnan(np.sum(temp)):
            continue

        color_points = np.array(pts.colors)[sel_idx]
        color_points[:,[0,1,2]] = color_points[:,[2,1,0]]
        
        pts.points = o3d.utility.Vector3dVector(temp)
        pts.colors = o3d.utility.Vector3dVector(color_points)
        data = np.concatenate([temp,color_points], axis=1)
        
        o3d.io.write_point_cloud(sys.argv[2]+scene.split("/")[-1].split(".")[0]+"/"+frame+".ply", pts)
        np.save(sys.argv[3]+scene.split("/")[-1].split(".")[0]+"/"+frame+".npy", data)

        datalist.append(os.path.abspath(sys.argv[3]+scene.split("/")[-1].split(".")[0]+"/"+frame+".npy"))
        
        counter = 0
        success += 1

    print (success)

np.save(sys.argv[4], datalist)
