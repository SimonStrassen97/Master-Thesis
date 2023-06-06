#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:42:15 2023

@author: simonst
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
from pycolmap_utils.read_write_model import read_images_binary
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d



def load_data(data_path):
    
    with open(data_path, "r") as f:
        data = json.load(f)
        
    return data




def process_colmap_pcl(colmap_path, data_path):
    
    data_folder = os.path.join(data_path, "data")
    
    
    pcd_path = os.path.join(colmap_path, "dense.ply")
    
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    
    # colmap poses
    images_file = os.path.join(colmap_path, "sparse", "images.bin")
    images = read_images_binary(images_file)
    
    qw, qx, qy, qz = images[1].qvec
    Q = np.array([qx, qy, qz, qw])
    
    
    qw, qx, qy, qz = images[1].qvec
    Q = np.array([qx, qy, qz, qw])
    
    R = Rot.from_quat(Q).as_matrix()
    t_ = np.expand_dims(images[1].tvec,axis=-1)
    T_w2C = np.hstack([R,t_])
    T_w2C = np.vstack([T_w2C, [0,0,0,1]])
    
    
    pcd.transform(T_w2C)
    
    
    pts = np.array(pcd.points)
    colors = np.array(pcd.colors)
    normals = np.array(pcd.normals)
    
    t = np.zeros(6)
    for i in range(6):
        t1 = images[i+1].tvec
        t2 = images[i+2].tvec
        t[i] = (np.linalg.norm(t1-t2))
     
    
    d = t.mean()
    
    scaling = 0.045/d
    
    pts = pts*scaling
    pcd.points = o3d.utility.Vector3dVector(pts)
    
    files = os.listdir(data_folder)
    files.sort()
    
    data = load_data(os.path.join(data_folder, files[0]))
    c_w = data["cam"]
    
    R = Rot.from_euler("z", c_w[3]).as_matrix()
    t = np.expand_dims(np.array([c_w[0],c_w[1], c_w[2]]),axis=-1)/1000
    T_C2W_ = np.array(data["T_c2w"])
    T_C2W_[:3,3]/=1000
    T_C2W = np.hstack([R,t])
    
    T_C2W = np.vstack([T_C2W, [0,0,0,1]])
    
    pcd.transform(T_C2W_)

    points = np.array(pcd.points)

    border_x: tuple = (-0.01, 0.800)
    border_y: tuple = (-0.01, 0.600)
    border_z: tuple = (-0.01, 0.1575)

    x,y,z = border_x, border_y, border_z

    in_x = np.logical_and(points[:,0] > x[0], points[:,0] < x[1])
    in_y = np.logical_and(points[:,1] > y[0], points[:,1] < y[1])
    in_z = np.logical_and(points[:,2] > z[0], points[:,2] < z[1])
    
    condition = in_z & in_x & in_y
    ind = np.where(condition)[0]
     
    pcd = pcd.select_by_index(ind)
    # origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # o3d.visualization.draw_geometries([pcd, origin])
    
    o3d.io.write_point_cloud(os.path.join(data_path, "colmap_pcl.ply"), pcd)
    
    return pcd



# path = os.path.join("/home/simonst/github/pycolmap_out/", "wt1/mvs")
# data_path = "/home/simonst/github/Datasets/wt/raw/20230514_164433"

# pcd = process_colmap_pcl(path, data_path)
# origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
# o3d.visualization.draw_geometries([pcd, origin])
  





# x2 = []
# y2 = []
# z2 = []
# r2 = []
# ind2 = []

# for i, img in enumerate(images.values()):
#     qw, qx, qy, qz = img.qvec
#     Q = np.array([qx, qy, qz, qw])
#     t = img.tvec
    
#     if i==0:
        
#         R = Rot.from_quat(Q).as_matrix()
#         t_ = np.expand_dims(t,axis=-1)
#         T_p2c = np.hstack([R,t_])
#         T_p2c = np.vstack([T_p2c, [0,0,0,1]])
        
#         T_c2p = np.linalg.inv(T_p2c)
        
#         T_c2w = T_p2w @ T_c2p
    
#     t = T_c2w @ np.append(t,1)
#     a = Rot.from_matrix([T_c2w[:3,:3]]).as_euler("xyz")
    
    
#     x2.append(t[0])
#     y2.append(t[1])
#     z2.append(t[2])
#     # r.append(c_w[3])
#     # ind.append(i)
    
    



# # sensor poses
# files = os.listdir(data_folder)
# files.sort()

# x = []
# y = []
# z = []
# r = []
# ind = []

# for i, file in enumerate(files):
#     data_path = os.path.join(data_folder, file)
    
#     data = load_data(data_path)
#     c_w = data["cam"]
    
#     if i==0:
        
#         R = Rot.from_euler("z", c_w[3]).as_matrix()
#         t = np.expand_dims(np.array([c_w[0],c_w[1], c_w[2]]),axis=-1)
#         T_p2w = np.hstack([R,t])
#         T_p2w = np.vstack([T_p2w, [0,0,0,1]])

#     x.append(c_w[0])
#     y.append(c_w[1])
#     z.append(c_w[2])
#     r.append(c_w[3])
#     ind.append(i)
    
   

# fig, ax = plt.subplots(1,2)
# ax[0].scatter(x,y)
# ax[0].scatter(x2, y2)
# ax[1].scatter(ind, r)

    
    
    
    
    
    
    
    
    