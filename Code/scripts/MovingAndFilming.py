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
# import pandas as pd
import open3d as o3d


import pyrealsense2 as rs

from utils.camera_operations import StereoCamera
from utils.axis_operations import AxisMover
from utils.general import AxisConfigs, StreamConfigs, CalibParams

cv2.destroyAllWindows()


CAMERA = True
AXIS = True

vis=False
output_dir = "C:\\Users\SI042101\ETH\Master_Thesis\Data\PyData"


if AXIS:
    from PyTeMotion.AxisFunctions import Axis
    CGA_x = Axis("CGA", "x",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )
    CGA_r = Axis("CGA", "r",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )
    CGA_y = Axis("CGA", "y",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )
    CGA_z = Axis("CGA", "zshort",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )
    
    
    CGA_y.StartTeControl()

    
    CGA_z.Initialize()
    CGA_r.Initialize()
    CGA_x.Initialize()
    CGA_y.Initialize()
        

configs = AxisConfigs()
axis = AxisMover(CGA_x, CGA_y, CGA_z, CGA_r, configs)

calib_params = CalibParams
calibrated = axis.calibrate(calib_params, mode="r")

time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
data_output_folder = os.path.join(output_dir, time_string)
os.makedirs(data_output_folder)



#########################################################
# cali

# # Cam init
# stream_configs = StreamConfigs(c_hfov=848)
# stereo_cam = StereoCamera(activate_adv=True)
# stereo_cam.startStreaming(stream_configs)


# # Mover init
# configs = AxisConfigs(n_images=50)
# calibration = AxisMover(CGA_x, CGA_y, CGA_z, CGA_r, configs)


# ref = [285,200,250]
# calibration_points = np.array([(0,0,0,0),(0,100,0,0), (0,200,0,0), (0,300,0,0),
#                               (0,300,0,-45), (100,350,0,-45), (400, 400,0,-120),
#                               (450,0,0,120), (100,50,0,30), (200,0,0,60),
#                               (0,0,50,35), (200,0,50,75), (400,0,25,135),
#                               (150,350,75,-45), (300,400,60,-90), (500,200,60,180)])


# try:
#     while True:

        
#         # calc next move and execute
#         done = calibration.CalibrationMover(calibration_points)
                
#         # take img
#         color_img, _ = stereo_cam.getFrame(ret=True)
        
#         # saving imgs, pose data and pcl
#         stereo_cam.saveImages(calibration_folder, calibration.move_counter-1)
#         calibration.saveData(calibration_folder)
        
        
#         # Show images
#         if vis:
#             cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#             cv2.imshow('RealSense', color_img)
        
        
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#             if cv2.waitKey(0) & 0xFF == ord('c'):
#                 cv2.destroyAllWindows()
#                 continue
        
#         if done:
#             break

# finally:

#     # Stop streaming
#     stereo_cam.stopStreaming()
#     cv2.destroyAllWindows()



# ####################################################33
# #random Moves cali


# # Cam init
# stream_configs = StreamConfigs(c_hfov=848)
# stereo_cam = StereoCamera(activate_adv=True)
# stereo_cam.startStreaming(stream_configs)


# # Mover init
# configs = AxisConfigs(n_images=50)
# axis = AxisMover(CGA_x, CGA_y, CGA_z, CGA_r, configs)
# calibration = AxisMover(CGA_x, CGA_y, CGA_z, CGA_r, configs)


# try:
#     while True:

        
#         # calc next move and execute
#         done, dest = calibration.RandomCalibrationMoves()
                
#         # take img
#         cv2.waitKey(300)
#         color_img, _ = stereo_cam.getFrame(ret=True)
#         stereo_cam.saveImages(data_output_folder, calibration.move_counter-1)
#         calibration.saveData(data_output_folder)
        
        
#         # saving imgs, pose data and pcl
     
        
        
#         # Show images
#         if vis:
#             cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#             cv2.imshow('RealSense', color_img)
        
        
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 done = True
#                 break
#             if cv2.waitKey(0) & 0xFF == ord('c'):
#                 cv2.destroyAllWindows()
#                 continue
        
#         if done:
#             break

# finally:

#     # Stop streaming
#     stereo_cam.stopStreaming()
#     cv2.destroyAllWindows()
    
    
    


# #####################################################3
# # data acquisition

# Cam init
if CAMERA:
    stream_configs = StreamConfigs()
    stereo_cam = StereoCamera(activate_adv=True)
    stereo_cam.startStreaming(stream_configs, data_output_folder)
    

# Mover init
checkpoints = np.array([(0,0,0,0), (0,-1,0,270),
                          (-1,-1,0,180), (-1,0,0,90), (-1,0,0,90), (0,0,0,90)])


counter = 1
done = False
try:
    while True:

        
        # calc next move and execute
        if AXIS:
            done, target, pos = axis.MovePlanner(checkpoints, ret=True)
            axis.saveData(data_output_folder)
            axis.saveData_(data_output_folder, counter)
            counter = axis.move_counter
        
        time.sleep(1)
        # take img
        if CAMERA:
            img, d, d_avg = stereo_cam.getFrame(ret=True)
            
            # saving imgs, pose data and pcl
            stereo_cam.saveImages(data_output_folder, counter-1)
        # stereo_cam.savePCL(data_output_folder, counter-1)
        
        counter += 1
        
        
        
        # Show images
        if vis and CAMERA:
            # images = np.hstack([img, d])
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', img)
        
        
            if cv2.waitKey(0) & 0xFF == ord('q'):
                done = True
                break
            if cv2.waitKey(0) & 0xFF == ord('c'):
                cv2.destroyAllWindows()
                continue
        
        if done:
            break

finally:

    # Stop streaming
    if CAMERA:
        stereo_cam.stopStreaming()
        cv2.destroyAllWindows()
        


