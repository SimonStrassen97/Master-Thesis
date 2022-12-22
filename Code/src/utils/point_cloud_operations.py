# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:06:43 2022

@author: SI042101
"""

import numpy as np
import open3d as o3d



class PointCloud():
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.pcl = o3d.geometry.PointCloud()
        
    def registrationFast(self, img, dmap, K, scale=1):
    #bacically this is a vectorized version of depthToPointCloudPos()
        C, R = np.indices(dmap.shape)
        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]
    
        R = np.subtract(R, cx)
        R = np.multiply(R, dmap)
        R = np.divide(R, fx * scale)
    
        C = np.subtract(C, cy)
        C = np.multiply(C, dmap)
        C = np.divide(C, fy * scale)
        
        pts = np.column_stack((dmap.ravel() / scale, R.ravel(), -C.ravel()))
        colors = np.column_stack((img[:,:,0].ravel(), img[:,:,1].ravel(), img[:,:,2].ravel()))
        
        self.pcl.points = o3d.utility.Vector3dVector(pts)
        self.pcl.colors = o3d.utility.Vector3dVector(colors/255)
        
        
    def CamToWorld(self, cam_pose):
        
        self.pcl.transform(cam_pose)
            
        
    def visualize(self, coord_frame=True):
        
        if coord_frame:
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=np.array([0., 0., 0.]))
            
        o3d.visualization.draw_geometries([self.pcl, origin])  