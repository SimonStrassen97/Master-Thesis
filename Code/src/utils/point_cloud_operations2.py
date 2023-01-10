# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:06:43 2022

@author: SI042101
"""

import os
import copy
import numpy as np
import pandas as pd
import open3d as o3d
import cv2

from scipy.spatial.transform import Rotation as Rot



class PointCloud():
    def __init__(self, pcl_configs, cam_offset):
        # self.vis = o3d.visualization.Visualizer()
        self.pcl = o3d.geometry.PointCloud()
        self.pcl_ = o3d.geometry.PointCloud()
        self.pcls = []
        self.pcls_ = []
        
        self.imgs = []
        self.depths = []
        
        self.cam_offset = cam_offset
        self.configs = pcl_configs 
        self.pose_data = None
        self.idx_list = []
        
        self.outliers = []
        self.inliers = []
    
        
        
    def ProcessPCL(self):
          
        print("---------------------")
        for i,pcl in enumerate(self.pcls_):
            print(f"{i+1}/{len(self.pcls_)}")
            
            self.idx = self.idx_list[i]
            self.pcl = copy.deepcopy(pcl)
            
            self.CamToArm()
            # self.CleanUpPCL()
            self.pcls.append(self.pcl)
            
        if self.configs.vis:
            self.visualize(self.pcls,
                           coord_frame=self.configs.coord_frame,
                           coord_scale=self.configs.coord_scale,
                           outliers=self.configs.outliers,
                           color=self.configs.color)
            
                   
        
    def CamToWorld(self):
        
        pass
        
    
    def _PCLToCam(self):
            
            
        R  = self.pcl_.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        self.pcl_.rotate(R, center=(0,0,0))
    
        
    def CamToArm(self):
        
        # align cam coordinates with RGA coordinates
        # pitch yaw roll defined as in target coordinates/Arm coordinates
        
    
        # view direction is z coordinate in camera frame
        self.view_dir = [0,0,1]

       
        # rotation to arm coordinates
        pitch =  -(self.cam_offset.r_y + 90)
        yaw = -(self.cam_offset.r_z + 90 - self.pose_data[self.idx, 8])
        roll = -(self.cam_offset.r_x)
        
        R = Rot.from_euler("xzx", [pitch, yaw, roll], degrees=True)
        self.view_dir = R.apply(self.view_dir)
        self.pcl.rotate(R.as_matrix(), center=(0,0,0))
        
        
        # rotate offsets given in arm coordinates
        R = Rot.from_euler("z", self.pose_data[self.idx,8], degrees=True)
        offsets = (self.cam_offset.x, self.cam_offset.y, self.cam_offset.z)

        (offsets_x, offset_y, offset_z) = R.apply(offsets)
        
        # add up position and camera offset (both in mm)
        T_x = (offsets_x + self.pose_data[self.idx, 5])/1000
        T_y = (offset_y + self.pose_data[self.idx, 6])/1000
        T_z = (offset_z + self.pose_data[self.idx, 7])/1000
        T = (T_x, T_y, T_z)
        self.pcl.translate(T)
        
        
    def CleanUpPCL(self):
        
       
        outlier_cloud = self._removeBackground()
        outlier_cloud += self._removeOutliers()
        outlier_cloud += self._removeHiddenPts()

        
        # outlier.cloud.paint_uniform_color([1, 0, 0])

        self.outliers.append(outlier_cloud)
    
    
    def _removeHiddenPts(self):
              
        _, ind = self.pcl.hidden_point_removal(self.pose_data[self.idx,5:8]/1000, self.configs.hp_radius)
        
        outlier_cloud = self.pcl.select_by_index(ind, invert=True)
        self.pcl = self.pcl.select_by_index(ind)
        
        outlier_cloud.paint_uniform_color([1, 0, 0])
        
        return outlier_cloud
      
        
    def _removeOutliers(self):
        
        _, ind = self.pcl.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=1)
        
        # cl, self.ind = self.pcl_r.remove_radius_outlier(nb_points=5, radius=0.020)
        
        outlier_cloud = self.pcl.select_by_index(ind, invert=True)
        self.pcl = self.pcl.select_by_index(ind)
        
        outlier_cloud.paint_uniform_color([0.7, 0.7, 0])
    
        
        return outlier_cloud
     
        
        
    def _removeBackground(self):
        
        points = np.array(self.pcl.points)
         
        in_x = np.logical_and(points[:,0] > self.configs.border_x[0],
                                  points[:,0] < self.configs.border_x[1])
        
       
        in_y = np.logical_and(points[:,1] > self.configs.border_y[0],
                                  points[:,1] < self.configs.border_y[1])
        
        in_z = np.logical_and(points[:,2] < -self.configs.border_z[0],
                                  points[:,2] > -self.configs.border_z[1])
        
        
        condition = in_z & in_x & in_y
        
        
        ind = np.where(condition)[0]
        outlier_cloud = self.pcl.select_by_index(ind, invert=True)
        self.pcl = self.pcl.select_by_index(ind)
        
        outlier_cloud.paint_uniform_color([1, 0.7, 0])
        
        return outlier_cloud
    
    def registration(self):
        
        
        
        pass
        
    def createMesh(self):
        
        method = self.configs.recon_method
        pcl = o3d.geometry.PointCloud()        
        for p in self.pcls:
            pcl += p
        
        if method=="ball":
            radii = [0.1, 0.04, 0.04, 0.08]
            self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcl, o3d.utility.DoubleVector(radii))       
            
        if method=="alpha":
            
            self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcl, alpha=0.01)
        
        if method=="poisson":
                
            self.mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcl, depth=9)
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            self.mesh.remove_vertices_by_mask(vertices_to_remove)
         
    def loadPoseData(self, path):
        
        with open(os.path.join(path, "pose_info.csv"), "r") as f:
            self.pose_data = pd.read_csv(f).to_numpy()
        

    def loadPCL(self, path):
        
        self.loadPoseData(path)
        
        pcl_folder = os.path.join(path, "PCL")
        files = os.listdir(pcl_folder)
        
        if self.configs.n_images:
            every_x = int(len(files)/self.configs.n_images)
            start = int(every_x/2)
            files = files[start::every_x]  
        
        for file in files:
            fpath = os.path.join(pcl_folder, file)
            pcl = o3d.io.read_point_cloud(fpath)
            self.pcl_ = pcl.voxel_down_sample(voxel_size=self.configs.voxel_size)
            
            self._PCLToCam()
            self._limitDepth()
            
                
            self.idx_list.append(int(os.path.basename(fpath).split("_")[0]))  
            self.pcls_.append(self.pcl_)
            
    
    def loadPCLfromDepth(self, path, K, depth_scale):
        
        self.loadPoseData(path)
        
        depth_folder = os.path.join(path, "depth")
        dfiles = os.listdir(depth_folder)
        
        
        img_folder = os.path.join(path, "img")
        ifiles = os.listdir(img_folder)
        
        dpt_folder = os.path.join(path, "dpt_depth")
        dptfiles = os.listdir(dpt_folder)
        dptfiles = [d for d in dptfiles if d.endswith(".png")]
        
        if self.configs.n_images:
            every_x = int(len(dptfiles)/self.configs.n_images)
            start = int(every_x/2)
            dptfiles = dptfiles[start::every_x] 
            ifiles = ifiles[start::every_x]
            
            
        for dptfile, dfile, ifile in zip(dptfiles, dfiles, ifiles):
            dptpath = os.path.join(dpt_folder, dptfile)
            dpath = os.path.join(depth_folder, dfile)
            ipath = os.path.join(img_folder, ifile)
            
            dpt = cv2.imread(dptpath,0)
            depth = cv2.imread(dpath,0)
            img = cv2.imread(ipath)
            
            
            
            pcl_ = self.pcl_from_depth(img, depth, K)
            self._limitDepth()
            
            
            self.idx_list.append(int(os.path.basename(dptpath).split("_")[0]))  
            self.pcls_.append(pcl_)
            self.imgs.append(img)
            self.depths.append(depth)
            
        
        
        
    
    def _limitDepth(self):
        
        if self.configs.depth_thresh:
            points = np.array(self.pcl_.points)
            
            condition = points[:,2] < self.configs.depth_thresh
            ind = np.where(condition)[0]
            self.pcl_ = self.pcl_.select_by_index(ind)
            
        
        
    def pcl_from_depth(self, img, dmap, K, scale=1, depth_scale=1):
    #bacically this is a vectorized version of depthToPointCloudPos()
    
        
        R, C = np.indices(dmap.shape)
        fx = K[0,0]
        fy = K[1,1]
        cy = K[0,2]
        cx = K[1,2]
    
        R = np.subtract(R, cx)
        R = np.multiply(R, dmap)
        R = np.divide(R, fx * scale)
    
        C = np.subtract(C, cy)
        C = np.multiply(C, dmap)
        C = np.divide(C, fy * scale)
        
        # pts = np.column_stack((dmap.ravel()/scale, R.ravel(), -C.ravel()))
        pts = np.column_stack((C.ravel(), R.ravel(), dmap.ravel()/scale ))

        colors = np.column_stack((img[:,:,0].ravel(), img[:,:,1].ravel(), img[:,:,2].ravel()))
        
        self.pcl_.points = o3d.utility.Vector3dVector(pts)
        self.pcl_.colors = o3d.utility.Vector3dVector(colors/255)
        
        return self.pcl_
        

    def visualize(self, pcl_in, coord_frame=True, coord_scale=1, outliers=True, color=None):
        
        
        vis_list = []
        
        pcl_ = copy.deepcopy(pcl_in)
        
        if type(pcl_)==list:
            pcl = o3d.geometry.PointCloud()
            for p in pcl_:
                pcl += p
        else:
            pcl = pcl_
            
        
        if color:
            
            color_ = self._StringToColor(color)
            if color_:
                pcl.paint_uniform_color(color_)
        
        vis_list.append(pcl)
        
        if coord_frame:
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_scale, origin=np.array([0., 0., 0.]))
            vis_list.append(origin)
            
        if outliers:
            vis_list += self.outliers
            
        o3d.visualization.draw_geometries(vis_list)
        
    def _StringToColor(self, string):
        
        color = None
        
        if string=="gray":
            color = [0.8,0.8,0.8]
                    
        elif string=="green":
            color = [0,1,0]
        
        elif string=="blue":
            color = [0,0,1]
        
        elif string=="orange":
            color = [1,0.7,0]
            
        
        return color
            
        
    


