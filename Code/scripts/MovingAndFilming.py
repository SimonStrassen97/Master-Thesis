# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:53:17 2022

@author: SI042101
"""


import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import open3d as o3d


import pyrealsense2 as rs
from PyTeMotion.AxisFunctions import Axis
from utils.camera_operations import StereoCamera
from utils.axis_operations import AxisMover
from utils.general import AxisConfigs, StreamConfigs

cv2.destroyAllWindows()


output_dir = "C:\\Users\SI042101\ETH\Master_Thesis\Data\PyData"

time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
data_output_folder = os.path.join(output_dir, time_string)
os.makedirs(data_output_folder)

vis = False


CGA_x = Axis("CGA", "x",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )
CGA_r = Axis("CGA", "r",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )
CGA_y = Axis("CGA", "y",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )
CGA_z = Axis("CGA", "zshort",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )


CGA_y.StartTeControl()


CGA_z.Initialize()
CGA_r.Initialize()
CGA_x.Initialize()
CGA_y.Initialize()
    


# Cam init
stream_configs = StreamConfigs(c_hfov=848)
stereo_cam = StereoCamera(activate_adv=True)
stereo_cam.startStreaming(stream_configs)


# Mover init
configs = AxisConfigs(n_images=50)
axis_mover = AxisMover(CGA_x, CGA_y, CGA_z, CGA_r, configs)



checkpoints = np.array([(0,0,0,0), (0,-1,0,270),
                         (-1,-1,0,180), (-1,0,0,90), (-1,0,40,90), (0,0,40,90)])

try:
    while True:

        
        # calc next move and execute
        done, target_pos, error = axis_mover.MovePlanner(checkpoints, ret=True)
                
        # take img
        color_img, depth_img = stereo_cam.getFrame(ret=True)
        
        # saving imgs, pose data and pcl
        stereo_cam.saveImages(data_output_folder, axis_mover.move_counter)
        stereo_cam.savePCL(data_output_folder, axis_mover.move_counter)
        axis_mover.saveData(data_output_folder)
        
        
        # Show images
        if vis:
            images = np.hstack([color_img, depth_img])
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
        
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.waitKey(0) & 0xFF == ord('c'):
                cv2.destroyAllWindows()
                continue
        
        if done:
            break

finally:

    # Stop streaming
    stereo_cam.stopStreaming()
    cv2.destroyAllWindows()
    

# mesh = o3d.io.read_triangle_mesh("C:/Users\SI042101\ETH\Master_Thesis\Data\PyData/20221221_155120\PCL\pcl_frame_51.ply")   
# # pcl = o3d.io.read_point_cloud("C:/Users\SI042101\ETH\Master_Thesis\Data\PyData/20221221_145837\PCL\pcl_frame_1.ply")   
# # mesh.compute_vertex_normals()
  
    
# o3d.visualization.draw_geometries([mesh])

