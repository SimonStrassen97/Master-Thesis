# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 19:38:35 2022

@author: SI042101
"""

  

import os
import pandas as pd
import copy
import open3d as o3d
import numpy as np
import cv2


folder = "C:/Users\SI042101\ETH\Master_Thesis\Data/PyData/20221221_155120"
pcl_folder = "C:/Users\SI042101\ETH\Master_Thesis\Data/PyData/20221221_155120/PCL_old"
i_folder = "C:/Users\SI042101\ETH\Master_Thesis\Data/PyData/20221221_155120/img_old"
d_folder = "C:/Users\SI042101\ETH\Master_Thesis\Data/PyData/20221221_155120/depth_old"

out_f = "C:/Users\SI042101\ETH\Master_Thesis\Data/PyData/20221221_155120/PCL"
out_i = "C:/Users\SI042101\ETH\Master_Thesis\Data/PyData/20221221_155120/img"
out_d = "C:/Users\SI042101\ETH\Master_Thesis\Data/PyData/20221221_155120/depth"

# os.makedirs(out_f)
# os.makedirs(out_i)
# os.makedirs(out_d)


listf = os.listdir(pcl_folder)
listd = os.listdir(d_folder)
listi = os.listdir(i_folder)

    

for file in listf:
    path = os.path.join(pcl_folder, file)
    pcl = o3d.io.read_point_cloud(path)
    # mesh = o3d.io.read_triangle_mesh(path)
    
    aa = file[:-4].split("_")
    for i in aa:
        if i.isdigit():
            k = str(int(i)-1)
            idx = k.zfill(4)
            
    
    name = f"{idx}_pcl.ply"
    out = os.path.join(out_f, name)
    o3d.io.write_point_cloud(out, pcl)
    
    
    


for file in listd:
    path = os.path.join(d_folder, file)
    d = cv2.imread(path)
    
    
    aa = file[:-4].split("_")
    for i in aa:
        if i.isdigit():
            k = str(int(i)-1)
            idx = k.zfill(4)
    
    name = f"{idx}_depth.png"
    out = os.path.join(out_d, name)
    cv2.imwrite(out, d)
    


for file in listi:
    path = os.path.join(i_folder, file)
    img = cv2.imread(path)
    
    
    aa = file[:-4].split("_")
    for i in aa:
        if i.isdigit():
            k = str(int(i)-1)
            idx = k.zfill(4)
    
    name = f"{idx}_img.png"
    out = os.path.join(out_i, name)
    cv2.imwrite(out, img)
    
    
    
    