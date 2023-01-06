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

from utils.general import CameraOffset, PCLConfigs, StreamConfigs
from utils.point_cloud_operations2 import PointCloud
from utils.camera_operations import StereoCamera
import utils.dpt_monodepth as dpt




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


pcl = PointCloud(pcl_configs, cam_offset, path)

pcl.ProcessPCL()


pcl.createMesh()

o3d.visualization.draw_geometries([pcl.mesh])







#######################################################################################################




input_folder = os.path.join(path, "img")
output_folder = os.path.join(path, "dpt_depth")

weights_path = "J:/GitHub/DPT/weights/dpt_hybrid-midas-501f0c75.pt"
# model_type = "dpt_hybrid"
model_type = "dpt_hybrid_nyu"
optimize = True
absolute_depth = False




for name in os.listdir(input_folder):
    
    img = cv2.imread(os.path.join(input_folder, name))
    # depth = cv2.imread(os.path.join(output_folder, f"{orig[:-4]}.png"))
    # rect_img = cam.undistort(img)
    
    inv_depth = dpt.run(img, name, output_folder, weights_path, model_type=model_type, absolute_depth=absolute_depth)
    
    
    
    
# Cam init
stream_configs = StreamConfigs(c_hfov=848)
stereo_cam = StereoCamera(activate_adv=False)
stereo_cam.loadIntrinsics()

K_c, K_d = stereo_cam.getIntrinsicMatrix()


cam_offset = CameraOffset()

pcl_configs = PCLConfigs(voxel_size=0.02,
                          depth_thresh=1,
                          vis=True,
                          # color="gray",
                          n_images=10,
                          outliers=True,
                          hp_radius=100,
                          recon_method="poisson"
                          )


pcl = PointCloud(pcl_configs, cam_offset, path)


img = cv2.imread(os.path.join(input_folder,"0000_img.png"))
depth = cv2.imread(os.path.join(output_folder,"0000_img.png"),0)


pcl.pcl_from_depth(img, depth, K_c, scale=100)
pcl._PCLToCam()

pcl.visualize(pcl.pcl_)




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


