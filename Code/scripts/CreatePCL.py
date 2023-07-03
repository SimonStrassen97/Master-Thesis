# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 11:34:48 2022

@author: SI042101
"""


import time
import os

import sys

PENet_path = '/home/simonst/github/PENet'
if PENet_path not in sys.path:
    sys.path.append(PENet_path)

import numpy as np
import open3d as o3d
import copy
import cv2
import matplotlib.pyplot as plt
import scipy.optimize

from utils.general import StreamConfigs, PCLConfigs, StreamConfigs, OffsetParameters
from utils.general import loadIntrinsics 
from utils.point_cloud_operations import PointCloud
from utils.point_cloud_operations2 import PointCloud2

from utils.camera_operations import StereoCamera
from utils.worktable_operations import Object, Worktable

import time
    
# import pyransac3d as pyrsc

# from pcls


cp1 = "/home/simonst/github/results/no_sparsifier/pe_train/model_best.pth.tar"
cp2 = "/home/simonst/github/results/dots_sparsifier/pe_train/model_best.pth.tar"
cp3 = "/home/simonst/github/results/edge_sparsifier/pe_train/model_best.pth.tar"

# path = "/home/simonst/github/Datasets/wt/raw/20230508_153614"
path1 = "/home/simonst/github/Datasets/wt/raw/20230514_164433" # wt1
path2 = "/home/simonst/github/Datasets/wt/raw/20230514_173628" # wt2
path3 = "/home/simonst/github/Datasets/wt/raw/20230522_163051" # wt3
path = "/home/simonst/github/Datasets/wt/raw/20230522_140447" # wt4.2

# path = "C:/Users/SI042101/ETH/Master_Thesis/Data/PyData/20230417_174256"


v = 0.0

pcl_configs = PCLConfigs(outliers=False, 
                         verbose=False,
                         pre_voxel_size=v, 
                         voxel_size=v,
                         hp_radius=75,
                         angle_thresh=75,
                         std_ratio_stat=2,
                         nb_points_stat=50,
                         nb_points_box=6,
                         box_radius=2*v,
                         registration_method="none",
                         filters=True
                         )


#no
pcl = PointCloud(pcl_configs)
pcd = pcl.create_multi_view_pcl(path, n_images=24)
# pcl.visualize(pcd, coord_scale=0.25, outliers=False)
out_path = os.path.join(path, "orig")
# if not os.path.exists(out_path):
    # os.makedirs(out_path)
# pcl.safe_pcl(os.path.join(out_path, "orig.ply"))
# cv2.imwrite(os.path.join(out_path, "orig.png"), pcl.depths[0])

# start = time.time()
# pcl1 = PointCloud(pcl_configs)
# pcd1 = pcl1.create_multi_view_pcl(path, n_images=5, run_s2d=cp1)
# print(f"no_sparsifier took {time.time()-start}")
# # pcl1.visualize(pcd1, coord_scale=0.25, outliers=False)
# out_path = os.path.join(path, "no_sparsifier")
# # if not os.path.exists(out_path):
#     os.makedirs(out_path)
# cv2.imwrite(os.path.join(out_path, "no_sparsifier.png"), pcl1.depths[0])
# pcl1.safe_pcl(os.path.join(out_path, "no_sparsifier.ply"))


#dots
# start = time.time()
# pcl2 = PointCloud(pcl_configs)
# pcd2 = pcl2.create_multi_view_pcl(path, n_images=1, run_s2d=cp2)
# print(f"dots_sparsifier took {time.time()-start}")
# # pcl2.visualize(pcd2, coord_scale=0.25, outliers=False)
# out_path = os.path.join(path, "dots_sparsifier")
# if not os.path.exists(out_path):
#     os.makedirs(out_path)
# cv2.imwrite(os.path.join(out_path, "dots_sparsifier.png"), pcl2.depths[0])
# pcl2.safe_pcl(os.path.join(out_path, "dots_sparsififer..ply"))


# edge
# start = time.time()
# pcl3 = PointCloud(pcl_configs)
# pcd3 = pcl3.create_multi_view_pcl(path, n_images=5, run_s2d=cp3)
# print(f"edge_sparsifier took {time.time()-start}")
# # pcl3.visualize(pcd3, coord_scale=0.25, outliers=False)
# out_path = os.path.join(path, "edge_sparsifier")
# if not os.path.exists(out_path):
#     os.makedirs(out_path)
# cv2.imwrite(os.path.join(out_path, "edge_sparsifier.png"), pcl3.depths[0])
# pcl3.safe_pcl(os.path.join(out_path, "edge_sparsifier..ply"))




# pcl.visualize(pcd, coord_scale=0.25, outliers=False)
# pcl1.visualize(pcd1, coord_scale=0.25, outliers=False)
# pcl2.visualize(pcd2, coord_scale=0.25, outliers=False)
# pcl3.visualize(pcd3, coord_scale=0.25, outliers=False)



# pcd2.paint_uniform_color([1,0,1])


origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
o3d.visualization.draw_geometries([pcd, origin])




# def custom_draw(pcd):
    
#     def rotate(vis):
#         ctr = vis.get_view_control()
#         ctr.rotate(1.0,0.0)
#         return False
    
#     origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
#     o3d.visualization.draw_geometries_with_animation_callback([pcd,origin], rotate)

# custom_draw(pcd)
# custom_draw(pcd1)
# custom_draw(pcd2)
# custom_draw(pcd3)

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



