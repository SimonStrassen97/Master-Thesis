# -*- coding: utf-8 -*-

"""
Created on Sun Jan 22 17:06:54 2023

@author: SI042101
"""


import numpy as np
import open3d as o3d


import matplotlib.cm as cm
from matplotlib.colors import Normalize





def StringToColor( string):
    
    color = None
    
    if string=="gray":
        color = [0.8,0.8,0.8]
                
    elif string=="green":
        color = [0,1,0]
    
    elif string=="blue":
        color = [0,0,1]
    
    elif string=="orange":
        color = [1,0.7,0]
        
    elif string=="red":
        color = [1,0,0]
        
    return color




class Object():
    def __init__(self):
        
        self.mesh = o3d.geometry.TriangleMesh()
        # self.pcl = o3d.geometry.PointCloud()
        self.aabb = np.array([[0,0,0],[0,0,0]])
        self.extent = np.array([0,0,0])
        self.center = np.array([0,0,0])
        self.id = None
        self.color = None
    
    def create_from_aabb(self, p1, p2, color=None):
        
        
        
        # p1 /= 1000
        # p2 /= 1000
        
        self.aabb = np.vstack([p1,p2])
        self.extent = p2-p1
        self.center = p1 + self.extent/2
        self.color = color
        
        if type(color)==str:
            color_ = StringToColor(color)
            print(color_)
            self.color = color_
        
        
        self.mesh = o3d.geometry.TriangleMesh.create_box(width=self.extent[0],height=self.extent[1],depth=self.extent[2])
        self.mesh.translate(self.aabb[0])
        if color:  
            self.mesh.paint_uniform_color(self.color)
            
        self.mesh.compute_vertex_normals()
        
    def create_from_center(self, p1, extent, color=None):
        
        # p1 /= 1000
        # extent /= 1000
        
        self.center = p1
        self.extent = extent
        self.aabb = np.vstack([p1-extent/2,p1+extent/2])
        self.color = color
        
        if type(color)==str:
            color_ = StringToColor(color)
            self.color = color_
        
        self.mesh = o3d.geometry.TriangleMesh.create_box(width=self.extent[0],height=self.extent[1],depth=self.extent[2])
        self.mesh.translate(self.aabb[0])
        if self.color:
            self.mesh.paint_uniform_color(self.color)
        
    
        self.mesh.compute_vertex_normals()
            


class Worktable():
    def __init__(self):
        
        self.model  = []
    
    def add(self, obj_list):
        
        self.model += obj_list
        
    
    def gridify_wt(self, pts, grid_size=0.01):
        x_max = pts[:,0].max()
        x_min = pts[:,0].min()
        y_max = pts[:,1].max()
        y_min = pts[:,1].min()
        z_max = pts[:,2].max()
        z_min = pts[:,2].min()
        
        
        grid_size = grid_size
        x_grid = np.arange(round(x_max/grid_size)+1)*grid_size
        y_grid = np.arange(round(y_max/grid_size)+1)*grid_size
        
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        
        bounding_boxes = []
        objects = []
        
    
        
        norm = Normalize(vmin=0, vmax=0.15)
        cmap = cm.autumn
        
        
        mapping = cm.ScalarMappable(norm=norm, cmap=cmap)
        
        
        for x_lower in x_grid:
            for y_lower in y_grid:
                
                x_upper = x_lower + grid_size
                y_upper = y_lower + grid_size
                
                
                condition_x = np.logical_and((pts[:,0] < x_upper), (pts[:,0] > x_lower))
                condition_y = np.logical_and((pts[:,1] < y_upper), (pts[:,1] > y_lower))
                candidates = np.where(np.logical_and(condition_x, condition_y))[0]
                
                if candidates.size:
                    # z_min = pts[candidates, 2].min()
                    z_lower =-0.025
                    z_upper = pts[candidates, 2].max()
                    
                else:
                    z_lower=-0.025
                    z_upper=0.0
                    
                p1 = np.array([x_lower, y_lower, z_lower])
                p2 = np.array([x_upper, y_upper, z_upper])
                
                bounding_boxes.append((p1,p2))
                
        
                color = mapping.to_rgba(z_upper)[:3]
               
                obj = Object()
                obj.create_from_aabb(p1, p2, color=None)
                
                self.model.append(obj)
                 
        
    def visualize(self):
        
        vis_list = []
        mesh = o3d.geometry.TriangleMesh()
        for instance in self.model:
            mesh += instance.mesh
           
       	mesh.compute_vertex_normals()
            
        vis_list = [instance.mesh for instance in self.model]
        
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        vis_list.append(origin)
        
        o3d.visualization.draw_geometries([mesh])
        # o3d.visualization.draw_geometries(vis_list)
        
        
        
        
        

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        