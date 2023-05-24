# -*- coding: utf-8 -*-

"""
Created on Sun Jan 22 17:06:54 2023

@author: SI042101
"""


import numpy as np
import open3d as o3d


import matplotlib.cm as cm
from matplotlib.colors import Normalize
import time

from utils.WT_configs import *




def StringToColor(string):
    
    color = None
    
    if string=="gray":
        color = [0.3,0.3,0.3]
                
    elif string=="green":
        color = [0,1,0]
    
    elif string=="blue":
        color = [0,0,1]
    
    elif string=="orange":
        color = [1,0.7,0]
        
    elif string=="red":
        color = [1,0,0]
        
    elif string=="black":
        color = [0.1,0.1,0.1]
        
    elif string=="white":
        color = [1,1,1]
        
    return color



class Object():
    def __init__(self, info):
        
        
        self.mesh = o3d.geometry.TriangleMesh()
        
        self.scale=1
        self.color="gray"
        
        if info["scale"]=="mm":
            self.scale = 1000
        self.size = np.array(info["size"])/self.scale
        self.name = info["name"]
        self.color = info["color"]

        self.aabb = None
        self.center = None
   
    def set_size(self, size):
        
        self.size = np.array(size)/self.scale
        
    def set_center(self, pos):
        
        self.center = np.array(pos)/self.scale
        
        if isinstance(self.size, np.ndarray):
            self.aabb = self.aabb = np.vstack([pos-self.size/2,pos+self.size/2])
            
    def set_color(self, color):
        
        self.color=color
    
    def set_corner(self, pos):
        
        pos = np.array(pos)/self.scale
        
        if isinstance(self.size, np.ndarray):
            self.aabb = np.vstack([pos, pos+self.size])
            self.center = pos+self.size/2
        
        else:
            print("Set size first.")
            

            
    def create_obj(self):
        
        if isinstance(self.size, np.ndarray):
            self.mesh = o3d.geometry.TriangleMesh.create_box(width=self.size[0],
                                                             height=self.size[1],
                                                             depth=self.size[2])
            self.mesh.compute_vertex_normals()
    

            if isinstance(self.aabb, np.ndarray):
                self.mesh.translate(self.aabb[0])
            
            elif isinstance(self.center, np.ndarray):
                self.mesh.translate(self.center-self.size/2)
                
            if type(self.color)==str:
                color = StringToColor(self.color)
                self.mesh.paint_uniform_color(color)
                
            elif type(self.color) in (tuple, list):
                self.mesh.paint_uniform_color(self.color)
    
            


class Worktable():
    def __init__(self):
        
        self.model  = []
        self.ref = []
        self.recon = []
        
        self.diff = []
        
        self.meshes = []
        self.model_mesh = o3d.geometry.TriangleMesh()
        self.recon_mesh = o3d.geometry.TriangleMesh()
        self.ref_mesh = o3d.geometry.TriangleMesh()
        self.diff_mesh = o3d.geometry.TriangleMesh()
        
        self.ref_pcl =  o3d.geometry.PointCloud()
        self.recon_pcl = o3d.geometry.PointCloud()
        
        self.grid_size=  None
        
    
    def create_model(self, wt_dict):
        for obj_name, pos in wt_dict.items():
        
            if "#" in obj_name:
                obj_name, _ = obj_name.split("#")
            obj_info = eval(obj_name)
        
            obj = Object(obj_info)
            obj.set_corner(pos)
            obj.create_obj()
           
            self.add_item(obj)
            # print(f"Adding {obj_name} at position {pos}")
        
    
    def add_item(self, obj):
        
        
        if type(obj)==list:
            self.model += obj
        
        elif type(obj)==Object:
            self.model.append(obj)
            
        else:
            raise ValueError
    
    def gridify_pcl(self, pcl):
        
        
        collection = []
        
        ref_pts = np.array([])
        if self.ref_pcl.points:
            ref_pts = np.asarray(self.ref_pcl.points)
        
        pts = np.asarray(pcl.points)
        x_max = pts[:,0].max()
        x_min = pts[:,0].min()
        y_max = pts[:,1].max()
        y_min = pts[:,1].min()
        z_max = pts[:,2].max()
        z_min = pts[:,2].min()
        
        
        x_grid = np.arange(round(x_max/self.grid_size)+1)*self.grid_size
        y_grid = np.arange(round(y_max/self.grid_size)+1)*self.grid_size
        
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        objects = []
        
        norm = Normalize(vmin=0, vmax=0.1)
        cmap = cm.autumn
        color_mapping = cm.ScalarMappable(norm=norm, cmap=cmap)

        for x_lower in x_grid:
            for y_lower in y_grid:
                
                x_upper = x_lower + self.grid_size
                y_upper = y_lower + self.grid_size
                
                
                # check if grid cell is empty or not
                in_xy=True
                if len(ref_pts):
        
                    # in_x = (ref_pts[:,0] < x_upper).max() &  (ref_pts[:,0] > x_lower).max()
                    # in_y = (ref_pts[:,1] < y_upper).max() & (ref_pts[:,1] > y_lower).max()
                    # in_xy = in_x & in_y
                    
                    condition_x = np.logical_and((ref_pts[:,0] < x_upper), (ref_pts[:,0] > x_lower))
                    condition_y = np.logical_and((ref_pts[:,1] < y_upper), (ref_pts[:,1] > y_lower))
                    candidates = np.where(np.logical_and(condition_x, condition_y))[0]
                    if not candidates.size:
                        in_xy = False
                    
                    
               
                if not in_xy:
                    continue
               
                else:
                    condition_x = np.logical_and((pts[:,0] < x_upper), (pts[:,0] > x_lower))
                    condition_y = np.logical_and((pts[:,1] < y_upper), (pts[:,1] > y_lower))
                    candidates = np.where(np.logical_and(condition_x, condition_y))[0]
                    
                            
                    if candidates.size:
                        # z_min = pts[candidates, 2].min()
                        z_lower =-0.004
                        z_upper = pts[candidates, 2].max() if pts[candidates, 2].max()>0 else 0
                    
                    
                    # else:
                    #     z_lower =-0.004
                    #     z_upper = 0
                    
                        p1 = np.array([x_lower, y_lower, z_lower])
                        p2 = np.array([x_upper, y_upper, z_upper])
                                        
                        color = color_mapping.to_rgba(z_upper)[:3]
                        
                        obj_info = {
                            "size": [self.grid_size,self.grid_size, z_upper-z_lower],
                            "name": "Grid_cell",
                            "color": color,
                            "scale": "m"
                            }
                        
                        obj = Object(obj_info)
                        obj.set_corner(p1)
                        obj.create_obj()
                        
                        objects.append(obj)
                                    
                
        return objects
                
                
    def get_ref_wt(self, grid_size=0.01, n_pts=250000):
        
        if not self.model:
            
            print("Not model to build reference.")
            pass
        
        if not self.model_mesh.vertices:
            self.compile_mesh()
        
        if not self.grid_size:
            self.grid_size=grid_size
        
        pcl = self.mesh_to_pcl(self.model_mesh, n_pts=n_pts)

        objs = self.gridify_pcl(pcl)
        
        self.ref  += objs
        self.ref_pcl = pcl
        
        if self.recon_pcl.points:
            self.crop_recon_pcl()
                
    def get_recon_wt(self, pcl, grid_size=0.01):
        
        self.recon_pcl = pcl
        
        if not self.grid_size:
            self.grid_size=grid_size
            
        objs = self.gridify_pcl(pcl)

        self.recon += objs
        
        if self.ref_pcl.points:
            self.crop_recon_pcl()
                 
        
    def compile_mesh(self):
        
        if not self.model_mesh.vertices:
            for instance in self.model:
                self.model_mesh += instance.mesh
                
            self.meshes.append(self.model_mesh)
            
        
        if not self.recon_mesh.vertices:
            for instance in self.recon:
                self.recon_mesh += instance.mesh
            
            self.meshes.append(self.recon_mesh)
        
            
        if not self.ref_mesh.vertices:
            for instance in self.ref:
                self.ref_mesh += instance.mesh
            
            self.meshes.append(self.ref_mesh)
            
        if not self.diff_mesh.vertices:
            for instance in self.diff:
                self.diff_mesh += instance.mesh
            
            self.meshes.append(self.diff_mesh)
            
            
    def crop_recon_pcl(self):
        
        ref_pts = np.array(self.ref_pcl.points)
        recon_pts = np.array(self.recon_pcl.points)
        colors = np.array(self.recon_pcl.colors)
        normals = np.array(self.recon_pcl.normals)
    
        idx = []
        for obj in self.model:
        
            
            if obj.size[2] > 0.005:
                continue
            
            aabb = obj.aabb
            
            condition = np.logical_and(recon_pts[:,1] > aabb[1,1], recon_pts[:,0] < aabb[1,0])
            idx += np.where(condition)[0].tolist()
            # if len(idx):
            #   pcl = pcl.select_by_index(idx, invert=True)
            
        
        new_pts = np.delete(recon_pts, idx, 0)
        new_normals = np.delete(normals, idx, 0)
        new_colors = np.delete(colors, idx, 0)
        
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(new_pts)
        pcl.colors = o3d.utility.Vector3dVector(normals)
        pcl.normals = o3d.utility.Vector3dVector(colors)
        
        x_max, _,  _ = ref_pts.max(axis=0)
        recon_pts = np.array(pcl.points)
        idx = np.where(recon_pts[:,0]<=x_max)[0]
        pcl = pcl.select_by_index(idx)
        
        self.recon_pcl = pcl
                
    def mesh_to_pcl(self, mesh, n_pts=1000000):
        
        pcl = mesh.sample_points_uniformly(number_of_points=n_pts)
        # pcl = mesh.sample_points_poisson_disk(number_of_points=n_pts)
        
        return pcl
    
    def evaluate_pcl(self):
    
        dist_recon2ref = self.recon_pcl.compute_point_cloud_distance(self.ref_pcl)
        dist_ref2recon = self.ref_pcl.compute_point_cloud_distance(self.recon_pcl)
    
        ae1 = np.array(dist_recon2ref)
        ae2 = np.array(dist_ref2recon)
        
        se1 = np.square(ae1)
        se2 = np.square(ae2)
        
        mae1 = ae1.mean()
        mae2 = ae2.mean()
        
        mse1 = se1.mean()
        mse2 = se2.mean()
        
        rmse1 = np.sqrt(mse1)
        rmse2 = np.sqrt(mse2)
        
        cd = np.sum(se1) + np.sum(se2)
        mcd = cd/(len(ae1)+len(ae2))
        
        recon2ref = {"mae": mae1, "mse": mse1, "rmse": rmse1}
        ref2recon = {"mae": mae2, "mse": mse2, "rmse": rmse2}
    
        return cd, mcd, recon2ref, ref2recon
            
        
    
    
    
    

    def evaluate_grids(self):
        
        
        z_ref_ = []
        z_recon_ = []
        z_diff_ = []
    
        if not self.recon:
            self.get_ref_wt()
        
        if not self.ref:
            raise ValueError("No reference")
            
            
        norm = Normalize(vmin=-0.1, vmax=0.1)
        cmap = cm.autumn
        cmap = cm.viridis
        color_mapping = cm.ScalarMappable(norm=norm, cmap=cmap)
        
        
        recon_idx = 0
        for ref in self.ref:
            
            ref_aabb = ref.aabb
            recon_aabb = self.recon[recon_idx].aabb
            
            current_cell = ref_aabb[:,:2]
            
            if np.all(current_cell==recon_aabb[:,:2]):
                
                z_ref = ref_aabb[1,2]
                z_recon = recon_aabb[1,2]
                z_diff = abs(z_recon - z_ref)
                
                z_ref_.append(z_ref)
                z_recon_.append(z_recon)
                z_diff_.append(z_diff)
                
                if z_diff > 0:
                    color = color_mapping.to_rgba(z_diff)[:3]
                    
                    obj_info = {
                        "size": [self.grid_size,self.grid_size, z_diff],
                        "name": "Grid_cell",
                        "color": color,
                        "scale": "m"
                        }
                    
                    obj = Object(obj_info)
                    if z_ref<z_recon:
                        obj.set_corner([ref_aabb[0,0], ref_aabb[0,1], 0])
                    else:
                        obj.set_corner([ref_aabb[0,0], ref_aabb[0,1], -z_diff])
                    obj.create_obj()
                    
                    self.diff.append(obj)
                    
                recon_idx += 1
                
            else: 
                
                
                z_ref = ref_aabb[1,2]
                z_recon = 0
                z_diff = abs(z_recon - z_ref)
                
                z_ref_.append(z_ref)
                z_recon_.append(z_recon)
                z_diff_.append(z_diff)
                
                z_diff = ref_aabb[1,2]
                if z_diff > 0:
                    color = color_mapping.to_rgba(z_diff)[:3]
        
                    obj_info = {
                        "size": [self.grid_size,self.grid_size, z_diff],
                        "name": "Grid_cell",
                        "color": color,
                        "scale": "m"
                        }
                    
                    obj = Object(obj_info)
                    obj.set_corner(ref_aabb[0])
                    obj.create_obj()
                    
                    self.diff.append(obj)
                    
        return z_ref_, z_recon_, z_diff_
        
                
        
    def visualize(self, model=False, recon=False, ref=False, diff=False):
        
        vis_list = []
        self.compile_mesh()
            
        if model:
            vis_list.append(self.meshes[0])
        
        if recon:
            vis_list.append(self.meshes[1])
        
        if ref:
            vis_list.append(self.meshes[2])
            
        if diff:
            vis_list.append(self.meshes[3])
        
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        vis_list.append(origin)
        
        # o3d.visualization.draw_geometries([self.mesh])
        o3d.visualization.draw_geometries(vis_list)
        
        
        
        
        

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        