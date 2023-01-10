# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 11:34:48 2022

@author: SI042101
"""


import time
import os

import numpy as np
import open3d as o3d
import copy
import cv2
import matplotlib.pyplot as plt
import scipy.optimize

from utils.general import CameraOffset, PCLConfigs, StreamConfigs
from utils.point_cloud_operations2 import PointCloud
from utils.camera_operations import StereoCamera
import utils.dpt_monodepth as dpt



# from pcls

path = "C:/Users\SI042101\ETH\Master_Thesis\Data/PyData/20221221_155120"


cam_offset = CameraOffset()

pcl_configs = PCLConfigs(voxel_size=0.005,
                          depth_thresh=1,
                          vis=True,
                          # color="gray",
                          n_images=10,
                          outliers=True,
                          hp_radius=100,
                          recon_method="poisson"
                          )


pcl = PointCloud(pcl_configs, cam_offset)
pcl.loadPCL(path)
pcl.ProcessPCL()


pcl.createMesh()

o3d.visualization.draw_geometries([pcl.mesh])







#######################################################################################################


# from images


input_folder = os.path.join(path, "img")
output_folder = os.path.join(path, "dpt_depth")
weights_path = "J:/GitHub/DPT/weights/dpt_hybrid-midas-501f0c75.pt"



# Cam init
stream_configs = StreamConfigs(c_hfov=848)
stereo_cam = StereoCamera(activate_adv=False)
stereo_cam.loadIntrinsics()

# stereo_cam.run_dpt(input_folder, output_folder, weights_path)

K_c, K_d = stereo_cam.getIntrinsicMatrix()

depth_scale = stereo_cam.intrinsics.get("depth").get("depth_scale")






cam_offset = CameraOffset()

pcl_configs = PCLConfigs(voxel_size=0.02,
                          depth_thresh=1,
                          vis=False,
                          # color="gray",
                          n_images=10,
                          outliers=True,
                          hp_radius=100,
                          recon_method="poisson"
                          )


pcl = PointCloud(pcl_configs, cam_offset)

pcl.loadPCLfromDepth("C:/Users\SI042101\ETH\Master_Thesis\Data\PyData/20221221_155120", K_c, depth_scale)
pcl.ProcessPCL()

pcl.visualize(pcl.pcls, coord_scale=100)

for p in pcl.pcls_:
    pcl.visualize(p, coord_scale=100)
    
    
    
    
    
    
stereo_cam.startStreaming(stream_configs)
stereo_cam.getFrame()

stereo_cam.saveImages(".", 0)
stereo_cam.run_dpt("./img", ".", weights_path)
stereo_cam.stopStreaming()


dpt = cv2.imread("./dpt/0000_dpt.png",-1)
depth =  cv2.imread("./depth/0000_depth.png",-1)
img = cv2.imread("./img/0000_img.png",-1)

true_depth = depth*depth_scale


def fun(x):
    f = 1/dpt.size * np.sum((dpt[depth>0]/x - true_depth[depth>0])**2)
    return f

ret = scipy.optimize.minimize(fun, 1)

dmap = dpt/ret.x

# dmap *= depth_scale

pcl.pcl_from_depth(img, dmap, K_c)
pcl.visualize(pcl.pcl_, coord_scale=0.2)

p = pcl.pcl_
p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=2))

o3d.visualization.draw_geometries([p])


fig, ax = plt.subplots(1,2)

ax[0].imshow(depth)
ax[1].imshow(dpt)


### code for additional clean up if needed


# for i in range(40,50):
#     pcl.visualize(pcl.pcls[i], outliers=False)
   
# points = [[0, 0, 0], list(pcl.view_dir2)]
# lines = [[0, 1]]
# colors = [[1, 0, 0] for i in range(len(lines))]
# line_set = o3d.geometry.LineSet()
# line_set.points = o3d.utility.Vector3dVector(points)
# line_set.lines = o3d.utility.Vector2iVector(lines)
# line_set.colors = o3d.utility.Vector3dVector(colors)

# p = pcl.pcls[0]
# p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# o3d.visualization.draw_geometries([p])

# n = np.asarray(p.normals)
# angles = np.arccos(np.dot(n,pcl.view_dir2))*180/np.pi



# _, pt_map = p.hidden_point_removal(pcl.pose_data[pcl.idx,5:8]/1000, 125)

# print("Visualize result")
# good = p.select_by_index(pt_map)
# bad  = p.select_by_index(pt_map, invert=True)



# # ind = np.where(angles<60)[0]

# # good = p.select_by_index(ind, invert=True)
# # bad = p.select_by_index(ind)

# good.paint_uniform_color([0.8, 0.8,0.8])
# bad.paint_uniform_color([1,0,0])
# origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=np.array([0., 0., 0.]))
# o3d.visualization.draw_geometries([good, bad, origin, line_set])


