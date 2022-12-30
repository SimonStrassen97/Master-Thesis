# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 11:34:48 2022

@author: SI042101
"""


import time
import numpy as np
import open3d as o3d
import copy

from utils.general import CameraOffset, PCLConfigs
from utils.point_cloud_operations import PointCloud




path = "C:/Users\SI042101\ETH\Master_Thesis\Data/PyData/20221221_155120"


cam_offset = CameraOffset()

pcl_configs = PCLConfigs(voxel_size=0.02,
                         depth_thresh=1,
                         vis=True,
                         #color="gray",
                         n_images=0,
                         outliers=False
                         )


pcl = PointCloud(pcl_configs, cam_offset, path)

pcl.ProcessPCL()

for i in range(10,25):
    pcl.visualize(pcl.pcls[i])
   
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




##########################################################

# pcl = PointCloud(pcl_configs, cam_offset, path)

# start1 = time.time()

# for i in range(10):
#     pcl.ProcessPCL()

# end1 = time.time()


# ########################################################3

# pcl2 = PointCloud2(pcl_configs, cam_offset, path)

# start2 = time.time()
# for i in range(10):
#     pcl.ProcessPCL()

# end2 = time.time()

# print(f"1:  {end1-start1}")
# print(f"2: {end2-start2}")

# pcl.visualize(outliers=False, color="gray")

# PCL = pcl.pcls[0]
# for p in pcl.pcls[1:]:
#     PCL += p
    
    
# import open3d as o3d

# o3d.visualization.draw_geometries([PCL])
