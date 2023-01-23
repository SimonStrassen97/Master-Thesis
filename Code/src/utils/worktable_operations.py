# -*- coding: utf-8 -*-

"""
Created on Sun Jan 22 17:06:54 2023

@author: SI042101
"""


import numpy as np
import open3d as o3d


x,y,z= 0.75,0.5,0.02
base = o3d.geometry.TriangleMesh.create_box(width=x,height=y,depth=z)
base.paint_uniform_color([1,0.8,0])

x,y,z= 0.1,0.1,0.1
box =  o3d.geometry.TriangleMesh.create_box(width=x,height=y,depth=z)
box.paint_uniform_color([0,0.8,1])

origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
elements = [origin, box, base]
o3d.visualization.draw_geometries(elements)

box.get_oriented_bounding_box()



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
        
    return color




class Object():
    def __init__(self):
        
        self.mesh = o3d.geometry.TriangleMesh()
        self.aabb = np.array([[0,0,0],[0,0,0]])
        self.extent = np.array([0,0,0])
        self.center = np.array([0,0,0])
        self.id = None
        self.color = None
    
    def create_from_aabb(self, p1, p2, color=None):
        
        p1 /= 1000
        p2 /= 1000
        
        self.aabb = np.vstack([p1,p2])
        self.extent = p2-p1
        self.center = p1 + self.extent/2
        color_ = StringToColor(color)
        self.color = color_
        
        self.mesh = o3d.geometry.TriangleMesh.create_box(width=self.extent[0],height=self.extent[1],depth=self.extent[2])
        self.mesh.translate(self.aabb[0])
        if self.color:  
            self.mesh.paint_uniform_color(self.color)
        
    def create_from_center(self, p1, extent, color=None):
        
        p1 /= 1000
        extent /= 1000
        
        self.center = p1
        self.extent = extent
        self.aabb = np.vstack([p1-extent/2,p1+extent/2])
        color_ = StringToColor(color)
        self.color = color_
        
        self.mesh = o3d.geometry.TriangleMesh.create_box(width=self.extent[0],height=self.extent[1],depth=self.extent[2])
        self.mesh.translate(self.aabb[0])
        if self.color:
            self.mesh.paint_uniform_color(self.color)
        
    
   
            

class Worktable():
    def __init__(self):
        
        self.model  = []
    
    def add(self, obj_list):
        
        self.model += obj_list
    
    def visualize(self):
        
        vis_list = []
        vis_list = [instance.mesh for instance in self.model]
        
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        vis_list.append(origin)
        
        o3d.visualization.draw_geometries(vis_list)
        
        
        
        
        

nest = Object()

center = np.array([270.8,247.5,34.5], dtype=float)
extent = np.array([137.3,95.0,69.0], dtype=float)
nest.create_from_center(center, extent, color="orange")


base = Object()

p1 = np.array([0,0,-5], dtype=float)
p2 = np.array([700, 500, 0], dtype=float)
base.create_from_aabb(p1, p2)





wt = Worktable()
wt.add([nest, base])

wt.visualize()
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        