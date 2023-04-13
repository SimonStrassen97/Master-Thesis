# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 11:34:48 2022

@author: SI042101
"""


import sys
path_to_model = '/home/simonst/github/PENet'
if path_to_model not in sys.path:
    sys.path.append(path_to_model)
    
    
import time
import os

import numpy as np
import open3d as o3d
import copy
import cv2
import matplotlib.pyplot as plt
import scipy.optimize

from utils.general import StreamConfigs, PCLConfigs, StreamConfigs, OffsetParameters
from utils.general import loadIntrinsics 
from utils.point_cloud_operations2 import PointCloud
from utils.camera_operations import StereoCamera
from utils.worktable_operations import Object, Worktable



# import pyransac3d as pyrsc

# from pcls


cp = "/home/simonst/github/results/pe_train/checkpoint-149.pth.tar"
path = "/home/simonst/github/Datasets/wt/raw/20230227_181401/"

cam_offset = OffsetParameters(r_z_cam=-30)
pcl_configs = PCLConfigs(outliers=False, 
                         voxel_size=0.001, 
                         n_images=4,
                         hp_radius=500,
                         angle_thresh=0,
                         std_ratio=10,
                         registration_method=None,
                         filters=False,
                         )

_,K,_ = loadIntrinsics()

# depth = cv2.imread(os.path.join(path, "depth", "0001_depth.png"), -1)
# out_size = (240,424)
# depth_, K_ = ResizeViaProjection(depth, K, out_size)

# inter = cv2.resize(depth, out_size[::-1], cv2.INTER_NEAREST)


# fig, ax = plt.subplots(1,2)
# ax[0].imshow(inter)
# ax[1].imshow(depth_)



pcl = PointCloud(pcl_configs, cam_offset)
pcl2 = PointCloud(pcl_configs, cam_offset)


# pcl.load_PCL_from_depth(path, K) 
pcl2.load_PCL_from_depth(path,K,run_s2d=cp)


# pcl.ProcessPCL()
pcl2.ProcessPCL()


# def custom_draw(pcd):
    
#     def rotate(vis):
#         ctr = vis.get_view_control()
#         ctr.rotate(1,0)
#         return False
    
#     o3d.visualization.draw_geometries_with_animation_callback([pcd], rotate)

# custom_draw(pcl.unified_pcl)

# p = pcl.unified_pcl
# pts = np.asarray(p.points)



         
# # gridified wt
   
# wt = Worktable()
# wt.gridify_wt(pts)
# wt.visualize()





# pts = np.asarray(p.points)
# normals = np.asarray(p.normals)
# labels = np.zeros((len(pts),1))

# output = np.hstack([pts, normals, labels])
# out_path = os.path.join(path, "pcl_out.txt")

# np.savetxt(out_path, output)

##############


# plane = pyrsc.Plane()
# _, ind = plane.fit(pts,0.01)

# base_ = p.select_by_index(ind)
# objects_ = p.select_by_index(ind, invert=True)

# _, ind = base_.remove_statistical_outlier(nb_neighbors=50, std_ratio=1)
# base = base_.select_by_index(ind)
# _, ind = objects_.remove_statistical_outlier(nb_neighbors=1000, std_ratio=0.0001)
# objects = objects_.select_by_index(ind)
# _, ind = objects.remove_radius_outlier(nb_points=75, radius=0.02)
# objects = objects.select_by_index(ind)

# correction = np.mean(np.asarray(base.points)[:,2])

# o3d.visualization.draw_geometries([base])
# o3d.visualization.draw_geometries([objects])


# p2_base = np.array(base.get_axis_aligned_bounding_box().max_bound)
# p1_base = np.array(base.get_axis_aligned_bounding_box().min_bound)
# p2_obj = np.array(objects.get_axis_aligned_bounding_box().max_bound)
# p1_obj = np.array(objects.get_axis_aligned_bounding_box().min_bound)



# # build model

# nest = Object()
# nest.create_from_aabb(p1_obj, p2_obj, color="orange")


# plate = Object()
# plate.create_from_aabb(p1_base, p2_base)


# wt = Worktable()
# wt.add([nest, plate])

# wt.visualize()
        


# # build reference wt

# ref_nest = Object()

# center = np.array([270.8,247.5,34.5], dtype=float)/1000 +np.array([0.075,0,0])
# extent = np.array([137.3,95.0,69.0], dtype=float)/1000
# ref_nest.create_from_center(center, extent, color="red")


# ref_base = Object()

# p1 = np.array([0,0,-5], dtype=float)/1000
# p2 = np.array([700, 500, 0], dtype=float)/1000
# ref_base.create_from_aabb(p1, p2)



# ref_wt = Worktable()
# ref_wt.add([ref_nest, ref_base])
# # ref_wt.visualize()

# test_wt = Worktable()
# test_wt.add([nest,plate,ref_nest])
# test_wt.visualize()
        
        






# pcl.createMesh()

# o3d.visualization.draw_geometries([pcl.mesh])

# mesh = pcl.mesh.filter_smooth_taubin()
# o3d.visualization.draw_geometries([mesh])


#######################################################################################################


# from images with ml


# input_folder = os.path.join(path, "img")
# output_folder = os.path.join(path, "dpt_depth")
# weights_path = "J:/GitHub/DPT/weights/dpt_hybrid-midas-501f0c75.pt"



# # Cam init
# stream_configs = StreamConfigs(c_hfov=848)
# stereo_cam = StereoCamera(activate_adv=False)
# stereo_cam.loadIntrinsics()

# # stereo_cam.run_dpt(input_folder, output_folder, weights_path)

# K_c, K_d = stereo_cam.getIntrinsicMatrix()

# depth_scale = stereo_cam.intrinsics.get("depth").get("depth_scale")






# cam_offset = CameraOffset()

# pcl_configs = PCLConfigs(voxel_size=0.02,
#                           depth_thresh=1,
#                           vis=False,
#                           # color="gray",
#                           n_images=10,
#                           outliers=True,
#                           hp_radius=100,
#                           recon_method="poisson"
#                           )


# pcl = PointCloud(pcl_configs, cam_offset)

# pcl.loadPCLfromDepth("C:/Users\SI042101\ETH\Master_Thesis\Data\PyData/20221221_155120", K_c, depth_scale)
# pcl.ProcessPCL()

# pcl.visualize(pcl.pcls, coord_scale=100)

# for p in pcl.pcls_:
#     pcl.visualize(p, coord_scale=100)
    
    
    
    
# stereo_cam.startStreaming(stream_configs)
# stereo_cam.getFrame()

# stereo_cam.saveImages(".", 0)
# stereo_cam.run_dpt("./img", ".", weights_path)
# stereo_cam.stopStreaming()


# dpt = cv2.imread("./dpt/0000_dpt.png",-1)
# depth =  cv2.imread("./depth/0000_depth.png",-1)
# img = cv2.imread("./img/0000_img.png",-1)

# true_depth = depth*depth_scale


# def fun(x):
#     f = 1/dpt.size * np.sum((dpt[depth>0]/x - true_depth[depth>0])**2)
#     return f

# ret = scipy.optimize.minimize(fun, 1)

# dmap = dpt/ret.x

# # dmap *= depth_scale

# pcl.pcl_from_depth(img, dmap, K_c)
# pcl.visualize(pcl.pcl_, coord_scale=0.2)

# p = pcl.pcl_
# p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=2))

# o3d.visualization.draw_geometries([p])


# fig, ax = plt.subplots(1,2)

# ax[0].imshow(depth)
# ax[1].imshow(dpt)



