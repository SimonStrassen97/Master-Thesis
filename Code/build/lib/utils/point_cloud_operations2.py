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
import torch

from scipy.spatial.transform import Rotation as Rot
from utils.general import ResizeWithAspectRatio


class PointCloud():
    def __init__(self, pcl_configs, offset_parameters):
        # self.vis = o3d.visualization.Visualizer()
        self.pcl = o3d.geometry.PointCloud()
        self.pcl_ = o3d.geometry.PointCloud()
        self.pcls = []
        self.pcls_ = []
        self.unified_pcl = o3d.geometry.PointCloud()
        
        self.imgs = []
        self.depths = []
        
        self.offsets = offset_parameters
        self.configs = pcl_configs 
        self.pose_data = None
        self.idx_list = []
        self.idx = None
        
        self.outliers = []
        self.inliers = []
    
        
        
    def ProcessPCL(self):
          
        for i,pcl in enumerate(self.pcls_):
            print("---------------------")
            print(f"{i+1}/{len(self.pcls_)}")
            print("---------------------")
            
            self.idx = self.idx_list[i]
            self.pcl = copy.deepcopy(pcl)
            
            self.CamToArm()
            self.CleanUpPCL()
            self.pcls.append(self.pcl)
            self.registration()
            
            self.unified_pcl += self.pcl
        
            
        self.unified_pcl = self.unified_pcl.voxel_down_sample(voxel_size=self.configs.voxel_size)
        self.ArmToWorld()
        # self.unified_pcl = self.unified_pcl.uniform_down_sample(4)
            
        if self.configs.vis:
            self.visualize(self.unified_pcl,
                           coord_frame=self.configs.coord_frame,
                           coord_scale=self.configs.coord_scale,
                           outliers=self.configs.outliers,
                           color=self.configs.color)
            
                   
        
    def ArmToWorld(self):
        
        
        T_x = (self.offsets.x_arm)/1000
        T_y = (self.offsets.y_arm)/1000
        T_z = (self.offsets.z_arm)/1000
        T = (T_x, T_y, T_z)
        
        self.unified_pcl.translate(T)
        
 
    def _PCLToCam(self):
            
            
        R  = self.pcl_.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        self.pcl_.rotate(R, center=(0,0,0))
    
        
    def CamToArm(self):
        
        # align cam coordinates with RGA coordinates
        # pitch yaw roll defined as in target coordinates/Arm coordinates
        
    
        # view direction is z coordinate in camera frame
        # self.view_dir = [0,0,1]
    
       
        # rotation to arm coordinates
        pitch =  -(self.offsets.r_y_cam + 90)
        yaw = -(self.offsets.r_z_cam + 90 - self.pose_data[self.idx, 8])
        roll = -(self.offsets.r_x_cam)
        
        R = Rot.from_euler("xzx", [pitch, yaw, roll], degrees=True)
        # self.view_dir = R.apply(self.view_dir)
        self.pcl.rotate(R.as_matrix(), center=(0,0,0))
        
        
        # rotate offsets given in arm coordinates
        R = Rot.from_euler("z", self.pose_data[self.idx,8], degrees=True)
        offsets = (self.offsets.x_cam, self.offsets.y_cam, self.offsets.z_cam)

        (offsets_x, offset_y, offset_z) = R.apply(offsets)
        
        # add up position and camera offset (both in mm)
        T_x = (offsets_x + self.pose_data[self.idx, 5])/1000
        T_y = (offset_y + self.pose_data[self.idx, 6])/1000
        T_z = -(offset_z + self.pose_data[self.idx, 7])/1000
        T = (T_x, T_y, T_z)
        self.pcl.translate(T)
        
        
        
    def CleanUpPCL(self):
        
        outlier_cloud = o3d.geometry.PointCloud()
        outlier_cloud += self._removeBackground()
        outlier_cloud += self._removeHiddenPts()
        outlier_cloud += self._removeInfeasablePts()
        outlier_cloud += self._removeOutliers()

        
        self.outliers.append(outlier_cloud)
        
    def _removeInfeasablePts(self):
        
        self.dir_matrix = np.asarray(self.pcl.points) - self.pose_data[self.idx, 5:8]/1000
        norm_c = np.linalg.norm(self.dir_matrix, axis=1)
        self.dir_matrix /= norm_c[:, np.newaxis]
        
        self.pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
        
        self.n = np.asarray(self.pcl.normals)
        self.angles = np.arccos(np.sum(self.n*self.dir_matrix, axis=1))*180/np.pi
        
    
        ind = np.where(self.angles>self.configs.angle_thresh)[0]
        
        print(f"View direction filter removed {len(self.pcl.points)-len(ind)} points.")
        outlier_cloud = self.pcl.select_by_index(ind, invert=True)
        self.pcl = self.pcl.select_by_index(ind)
        
        outlier_cloud.paint_uniform_color([0, 0, 1])
        
        return outlier_cloud
        

    
    def _removeHiddenPts(self):
              
        _, ind = self.pcl.hidden_point_removal(self.pose_data[self.idx,5:8]/1000, self.configs.hp_radius)
        print(f"Hidden Points filter removed {len(self.pcl.points)-len(ind)} points.")
        outlier_cloud = self.pcl.select_by_index(ind, invert=True)
        self.pcl = self.pcl.select_by_index(ind)
        
        outlier_cloud.paint_uniform_color([1, 0, 0])
        
        return outlier_cloud
      
        
    def _removeOutliers(self):
        
        _, ind = self.pcl.remove_statistical_outlier(nb_neighbors=50,
                                                    std_ratio=self.configs.std_ratio)
        
        
        print(f"Outlier filter removed {len(self.pcl.points)-len(ind)} points.")
        # cl, self.ind = self.pcl_r.remove_radius_outlier(nb_points=5, radius=0.020)
        
        outlier_cloud1 = self.pcl.select_by_index(ind, invert=True)
        self.pcl = self.pcl.select_by_index(ind)
        
        # _, ind = self.pcl.remove_radius_outlier(nb_points=self.configs.nb_points, radius=self.configs.outlier_radius)
        
        # print(f"Outlier filter removed {len(self.pcl.points)-len(ind)} points.")
        
        
        # outlier_cloud2 = self.pcl.select_by_index(ind, invert=True)
        # self.pcl = self.pcl.select_by_index(ind)
        
        
        outlier_cloud = outlier_cloud1 #+ outlier_cloud2
        outlier_cloud.paint_uniform_color([0.7, 0.7, 0])
    
        
        return outlier_cloud
     
        
        
    def _removeBackground(self):
        
        points = np.array(self.pcl.points)
        
        
        x,y,z = self.configs.border_x, self.configs.border_y, self.configs.border_z
        x_arm, y_arm, z_arm = self.offsets.x_arm/1000, self.offsets.y_arm/1000, self.offsets.z_arm/1000
        
        x = tuple(x_-x_arm for x_ in x)
        y = tuple(y_-y_arm for y_ in y)
        z = tuple(z_-z_arm for z_ in z)
        
        self.x=x
        self.y=y
        self.z=z
         
        in_x = np.logical_and(points[:,0] > x[0],
                                  points[:,0] < x[1])
        
       
        in_y = np.logical_and(points[:,1] > y[0],
                                  points[:,1] < y[1])
        
        in_z = np.logical_and(points[:,2] > z[0],
                                  points[:,2] < z[1])
        
        
        condition = in_z & in_x & in_y
        
        
        ind = np.where(condition)[0]
        print(f"Position filter removed {len(self.pcl.points)-len(ind)} points.")
        outlier_cloud = self.pcl.select_by_index(ind, invert=True)
        self.pcl = self.pcl.select_by_index(ind)
        
        outlier_cloud.paint_uniform_color([1, 0.7, 0])
        
        return outlier_cloud
    
    def registration(self):
        
        target = self.pcls[0]
        source = self.pcl
        
        current_transformation = np.identity(4)
    
    
        # source.paint_uniform_color((0,0,255))
        # target.paint_uniform_color((255,0,0))
        
        # point to plane
        if self.configs.registration_method == "plane":
            result_icp = o3d.pipelines.registration.registration_icp(
                source, target, self.configs.registration_radius, current_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            source.transform(result_icp.transformation)
            
        elif  self.configs.registration_method == "color":
            result_color_icp = o3d.pipelines.registration.registration_colored_icp(
                    source, target, self.configs.registration_radius, current_transformation)
            source.transform(result_color_icp.transformation)
           
        
    def createMesh(self):
        
        method = self.configs.recon_method
        
        if method=="ball":
            radii = [0.1, 0.04, 0.04, 0.08]
            self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(self.unified_pcl, o3d.utility.DoubleVector(radii))       
            
        if method=="alpha":
            
            self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(self.unified_pcl, alpha=0.01)
        
        if method=="poisson":
                
            self.mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.unified_pcl, depth=9)
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
            idx = np.linspace(0, len(files) - 1, self.configs.n_images).astype(int)
      
        
        for i in idx:
            fpath = os.path.join(pcl_folder, files[i])
            pcl = o3d.io.read_point_cloud(fpath)
            self.pcl_ = pcl.voxel_down_sample(voxel_size=self.configs.voxel_size)
            
            self._PCLToCam()
            self._limitDepth()
            
                
            self.idx_list.append(int(os.path.basename(fpath).split("_")[0]))  
            self.pcls_.append(self.pcl_)
            
    
    def loadPCLfromDepth(self, path, K, depth_scale, run_s2d=""):
        
        self.loadPoseData(path)
        
        depth_folder = os.path.join(path, "depth")
        dfiles = os.listdir(depth_folder)
        
        img_folder = os.path.join(path, "img")
        ifiles = os.listdir(img_folder)
        
        
        if self.configs.n_images:
            idx = np.linspace(0, len(dfiles) - 1, self.configs.n_images).astype(int)
            
            
        for i in idx:
            dpath = os.path.join(depth_folder, dfiles[i])
            ipath = os.path.join(img_folder, ifiles[i])
            
            depth = cv2.imread(dpath,-1)
            img = cv2.imread(ipath, -1)
            
            if run_s2d:
                checkpoint = torch.load(run_s2d)
                model = checkpoint["model"]
                model.eval()
                inp = self._prepareS2Dinput(img, depth)
                pred = model(inp)
                pred = pred.detach().cpu().numpy()
                
            
            pcl_ = self.pcl_from_depth(img, depth, K)
            self._limitDepth()
            
            
            self.idx_list.append(int(os.path.basename(dpath).split("_")[0]))  
            self.pcls_.append(pcl_)
            self.imgs.append(img)
            self.depths.append(depth)
            
    def _prepareS2Dinput(self, img, depth):
        
        
        rgb = ResizeWithAspectRatio(img, height=240)
        depth = ResizeWithAspectRatio(depth, height=240)
        rgb = np.asfarray(rgb, dtype='float') / 255
        depth = np.asfarray(depth, dtype="float") * 0.0001
        rgbd = np.append(rgb, np.expand_dims(depth, axis=2), axis=2)
        if rgbd.ndim == 3:
            rgbd = torch.from_numpy(rgbd.transpose((2, 0, 1)).copy())
        elif rgbd.ndim == 2:
            rgbd = torch.from_numpy(rgbd.copy())
        
        return rgbd.cuda()
        
        
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
            
        
    


