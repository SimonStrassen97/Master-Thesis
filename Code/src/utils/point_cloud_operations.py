# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:06:43 2022

@author: SI042101
"""

import os
import copy
import numpy as np
import open3d as o3d
import cv2
import torch
import json

from scipy.spatial.transform import Rotation as Rot
from model import PENet_C2

from utils.general import ResizeWithAspectRatio, Crop, ResizeViaProjection 
from utils.general import projectPoints, deprojectPoints
from utils.general import prepare_s2d_input, loadIntrinsics

torch.cuda.empty_cache()



class PointCloud():
    def __init__(self, pcl_configs):
        # self.vis = o3d.visualization.Visualizer()
        self.pcls_raw = []
        self.pcls = []
        self.unified_pcl = o3d.geometry.PointCloud()
        
        self.imgs = []
        self.depths = []
    
        self.configs = pcl_configs 
        
        self.K = None
        self.depth_scale = None
        self.pose_data = []
        
        self.idx_list = []
        self.idx = None
        
        self.outliers = []

    def load_data(self, dpath, ipath, data_path):
        
        
        depth = cv2.imread(dpath,-1)
        depth = depth * self.depth_scale
        
        img = cv2.imread(ipath, -1)
        img = img[...,::-1]
        
        with open(data_path, "r") as f:
            data = json.load(f)
            
        return depth, img, data
     
    def calc_pcl(self, img, depth, K):
    #bacically this is a vectorized version of depthToPointCloudPos()
    
        pts = deprojectPoints(depth, K, remove_zeros=False)
        colors = np.column_stack((img[:,:,0].ravel(), img[:,:,1].ravel(), img[:,:,2].ravel()))
        
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(pts)
        pcl.colors = o3d.utility.Vector3dVector(colors/255)
        
        points = np.asarray(pcl.points)
        condition = points[:,2]!=0
        ind = np.where(condition)[0]
        pcl = pcl.select_by_index(ind)
        
        pcl = self.limit_depth(pcl)
        
        return pcl
    
        
    def create_multi_view_pcl(self, path, n_images=5, run_s2d=""):
        
        
        if not self.K:
            self._loadIntrinsics(path)
            
        if run_s2d:
            torch.cuda.empty_cache()
            checkpoint = torch.load(run_s2d, map_location="cpu")
            args = checkpoint["args"]
            args.cpu =  True
            self.model = PENet_C2(args)
            self.model.load_state_dict(checkpoint['model'], strict=False) 
            self.model.eval()
            self.model.to("cpu")
            
        dfiles, ifiles, data_files = self._get_file_names(path, n_images)
        
        for n, (d, i, dd) in enumerate(zip(dfiles, ifiles, data_files)):
            
            print("-" * 20)
            print(f"{n+1}/{len(dfiles)}")
            print("-" * 20)
            
            depth, img, data = self.load_data(d,i,dd)
            K = self.K
        
            if run_s2d:
                
                print(f"Sparse to dense: {n+1}/{len(dfiles)}")
                img, _, _, depth, K_new = self.run_s2d(run_s2d, img, depth, K)
                K = K_new
            
             
            pcl = self.calc_pcl(img, depth, K)
            pcl = self.process_pcl(pcl, data)
                        
            self.imgs.append(img)
            self.depths.append(depth)
            self.pose_data.append(data)
            self.pcls.append(pcl)
            
            if n>0:
                pcl = self.registration(pcl)
        
            self.unified_pcl += pcl
            
        return self.unified_pcl
        
      
    
    def process_pcl(self, pcl, data):
        
        T_c2w = np.array(data["T_c2w"])
        T_c2w[:3,3] /= 1000  # mm to m
        cam_pose = np.array(data["cam"])
        cam_pose[:3] /= 1000 # mm to m
        
        pcl = self.cam_to_world(pcl, T_c2w)
        pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=10))
        
        if self.configs.filters:
            pcl, out1 = self.remove_background(pcl)
            pcl, out2 = self.remove_hidden_pts(pcl, cam_pose)
            pcl, out3 = self.remove_infeasable_pts(pcl, cam_pose)
            pcl, out4 = self.remove_outliers(pcl)
        
            outlier_cloud = out1 + out2, out3, out4
            self.outliers.append(outlier_cloud)
        
        return pcl
    
    def remove_background(self, pcl):
        
        points = np.array(pcl.points)
        x,y,z = self.configs.border_x, self.configs.border_y, self.configs.border_x
        
        in_x = np.logical_and(points[:,0] > x[0], points[:,0] < x[1])
        in_y = np.logical_and(points[:,1] > y[0], points[:,1] < y[1])
        in_z = np.logical_and(points[:,2] > z[0], points[:,2] < z[1])
        
        condition = in_z & in_x & in_y
        ind = np.where(condition)[0]
        print(f"Position filter removed {len(pcl.points)-len(ind)} points.")
        
        outlier_cloud = pcl.select_by_index(ind, invert=True)
        pcl = pcl.select_by_index(ind)

        outlier_cloud.paint_uniform_color([1, 0.7, 0])
        
        return pcl, outlier_cloud
    
    
    def remove_hidden_pts(self, pcl, cam_pose):
              
        _, ind = pcl.hidden_point_removal(cam_pose[:-1], self.configs.hp_radius)
        print(f"Hidden Points filter removed {len(pcl.points)-len(ind)} points.")
        outlier_cloud = pcl.select_by_index(ind, invert=True)
        pcl = pcl.select_by_index(ind)
        
        outlier_cloud.paint_uniform_color([1, 0, 0])
        
        return pcl, outlier_cloud
    
    def remove_outliers(self, pcl):
        
        _, ind = pcl.remove_statistical_outlier(nb_neighbors=50, std_ratio=self.configs.std_ratio)
        print(f"Outlier filter removed {len(pcl.points)-len(ind)} points.")
      
        outlier_cloud1 = pcl.select_by_index(ind, invert=True)
        pcl = pcl.select_by_index(ind)
        
        
        
        # _, ind = self.pcl.remove_radius_outlier(nb_points=self.configs.nb_points, radius=self.configs.outlier_radius)
        # print(f"Outlier filter removed {len(self.pcl.points)-len(ind)} points.")
        
        # outlier_cloud2 = self.pcl.select_by_index(ind, invert=True)
        # self.pcl = self.pcl.select_by_index(ind)
        
        
        outlier_cloud = outlier_cloud1 #+ outlier_cloud2
        outlier_cloud.paint_uniform_color([0.7, 0.7, 0])
    
        
        return pcl, outlier_cloud
    
    def remove_infeasable_pts(self, pcl, cam_pose):
        
        dir_matrix = np.asarray(pcl.points) - cam_pose[:-1]
        norm_c = np.linalg.norm(dir_matrix, axis=1)
        dir_matrix /= norm_c[:, np.newaxis]
                
        n = np.asarray(pcl.normals)
        angles = np.arccos(np.sum(n*dir_matrix, axis=1))*180/np.pi
        
    
        ind = np.where(angles>self.configs.angle_thresh)[0]
        
        print(f"View direction filter removed {len(pcl.points)-len(ind)} points.")
        outlier_cloud = pcl.select_by_index(ind, invert=True)
        pcl = pcl.select_by_index(ind)
        
        outlier_cloud.paint_uniform_color([0, 0, 1])

        return pcl, outlier_cloud
    
   
    def limit_depth(self, pcl):
        
        if self.configs.depth_thresh:
            points = np.array(pcl.points)
            
            condition = points[:,2] < self.configs.depth_thresh
            ind = np.where(condition)[0]
            pcl = pcl.select_by_index(ind)
            
        return pcl
    
    def run_s2d(self, model_path, img, depth, K):
        
        
        inp = self._prepare_s2d_input(img, depth, K)
        K_new = inp["K"].squeeze().numpy()
        img = (inp["rgb"].squeeze().numpy().transpose(1,2,0) * 255).astype(np.uint16)
        depth = inp["d"].squeeze().numpy()
        pred = self.model(inp)
        pred = pred.detach().cpu().squeeze().numpy()
        filled = depth.copy()
        filled[depth==0] = pred[depth==0]
        
        
        return img, depth, filled, pred, K_new
    
    def registration(self, pcl):
        
        target = self.pcls[0]
        source = pcl
        
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
            
        return pcl
    
    
    def create_mesh(self, pcl):
        
        method = self.configs.recon_method
        
        if method=="ball":
            radii = [0.1, 0.04, 0.04, 0.08]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcl, o3d.utility.DoubleVector(radii))       
            
        if method=="alpha":
            
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcl, alpha=0.01)
        
        if method=="poisson":
                
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcl, depth=9)
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
        return mesh
    
    
    def safe_pcl(self, path):
              
        o3d.io.write_point_cloud(path, self.unified_pcl)
        
            
    
    def cam_to_world(self, pcl, T_c2w):
    
        pcl = pcl.transform(T_c2w)
        
        return pcl
                      
    
    def _prepare_s2d_input(self, img, depth, K):
        
        return prepare_s2d_input(img, depth, K)
        
    def _loadIntrinsics(self, path):
        
        K_c, K_d, intr = loadIntrinsics(path)
        self.K = K_d
        self.depth_scale = intr["depth"]["depth_scale"]
        
        
    def _get_file_names(self, path, n_images):
        
        depth_folder = os.path.join(path, "depth")
        dfiles = os.listdir(depth_folder)
        dfiles.sort()
        
        idx = np.linspace(0, len(dfiles) - 1, n_images).astype(int)
        
        img_folder = os.path.join(path, "img")
        ifiles = os.listdir(img_folder)
        ifiles.sort()
        
        
        data_folder = os.path.join(path, "data")
        data_files = os.listdir(data_folder)
        data_files.sort()
        
        
        dfiles = [os.path.join(depth_folder, dfiles[x]) for x in idx]
        ifiles = [os.path.join(img_folder, ifiles[x]) for x in idx]
        data_files = [os.path.join(data_folder, data_files[x]) for x in idx]
        
        return dfiles, ifiles, data_files
    
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
            for x in self.outliers:
                vis_list += x
            
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
    
    
    
    
    


