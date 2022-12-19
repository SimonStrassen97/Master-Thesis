# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:53:17 2022

@author: SI042101
"""
import os
import sys

import cv2
import numpy as np

import pyrealsense2 as rs
# from PyTeMotion.AxisFunctions import Axis
from utils.camera_operations import StreamConfigs, StereoCamera


cv2.destroyAllWindows()
# CGA_x = Axis("CGA", "x",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )
# CGA_r = Axis("CGA", "r",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )
# CGA_y = Axis("CGA", "y",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )
# CGA_z = Axis("CGA", "zshort",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )


# CGA_y.StartTeControl()


# CGA_x.Initialize()
# CGA_y.Initialize()
# CGA_z.Initialize()
# CGA_r.Initialize()
    

# for x in [600, 1]: 
#     CGA_x.MoveTo(x)
#     print("------------------------------------------------")
#     print(CGA_x.GetCurrentPosition())
#     print("------------------------------------------------")
#     pass

# for y in [450, 1]:
#     CGA_y.MoveTo(y)
#     print("------------------------------------------------")
#     print(CGA_y.GetCurrentPosition())
#     print("------------------------------------------------")
#     pass

# for z in [200, 5]:
#     CGA_z.MoveTo(z)
#     print("------------------------------------------------")
#     print(CGA_z.GetCurrentPosition())
#     print("------------------------------------------------")
#     pass

# for r in [270, -180]: 
#     CGA_r.MoveFor(r)
#     print("------------------------------------------------")
#     print(CGA_r.GetCurrentPosition())
#     print("------------------------------------------------")
#     pass

# for r in [360]: 
#     CGA_r.MoveTo(r)
#     print("------------------------------------------------")
#     print(CGA_z.GetCurrentPosition())
#     print("------------------------------------------------")
#     pass



stream_configs = StreamConfigs(c_hfov=848)

stereo_cam = StereoCamera(activate_adv=True)

stereo_cam.startStreaming(stream_configs)
try:
    while True:

        color_img, depth_img = stereo_cam.getFrame()

        # Show images
        images = np.hstack([color_img, depth_img])
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:

    # Stop streaming
    stereo_cam.pipeline.stop()
    cv2.destroyAllWindows()





