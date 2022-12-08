# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:00:20 2022

@author: SI042101
"""
import os, sys
import glob


import cv2
import numpy as np
import matplotlib.pyplot as plt


# from PyTeMotion.AxisFunctions import Axis
from utils.camera_operations import Camera, StereoCamera, PseudoStereoCamera
from utils.general import ResizeWithAspectRatio




cv2.destroyAllWindows()
plt.close()



CHECKERBOARD = {}
CHECKERBOARD["size"] = (10,7)
CHECKERBOARD["scale"] = 25

cali_data_dir = "J:\GitHub\Recon\Data\CalibrationData/20221130_162627_calibration.json"

calibration_folder = os.path.realpath("C:/Users/SI042101/ETH/Master_Thesis/Images/Calibration_Test/")
input_folder = os.path.realpath("C:/Users/SI042101/ETH/Master_Thesis/Images/Stereo_Test/")

###################################################################################################

cam = Camera()

if cali_data_dir:
    cam.load_cam_params(cali_data_dir)
else:
    cam.calibrate(calibration_folder, CHECKERBOARD, save_data=True)

img =  cv2.imread(os.path.join(f"{input_folder}/left.png"),0)
rect_img = cam.undistort(img)


# fig1, ax1 = plt.subplots(1,2)
# ax1[0].imshow(img)
# ax1[1].imshow(rect_img)


####################################################################################################

stereo_cam = PseudoStereoCamera()

stereo_cam.load_cam_params(cali_data_dir)

img_l = cv2.imread(os.path.join(f"{input_folder}/left.png"),0)
# rect_img_l = cam.undistort(img_l, mode=0)

img_r= cv2.imread(os.path.join(f"{input_folder}/right.png"),0)
# rect_img_r = cam.undistort(img_r, mode=0)

stereo_cam.calibrate(input_folder, CHECKERBOARD)

rect_img_r, rect_img_l = stereo_cam.stereoUndistort(img_l, img_r)

stereo = cv2.StereoSGBM_create(numDisparities=64, blockSize=9)

disparity = stereo.compute(rect_img_l, rect_img_r)
# disparity = np.int16(disparity)

fig2, ax2 = plt.subplots(1,3)
ax2[0].imshow(rect_img_l)
ax2[1].imshow(rect_img_r)
ax2[2].imshow(disparity, "gray")






# ############################################


# cam2 = CameraMono()
# rel_depth = cv2.imread(test_dir,0).astype(np.uint8)
# cam2.load_cam_params(cali_data_dir)
# corrected_rel_depth = cam2.undistort(rel_depth)




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
#     print(CGA_z.GetCurrentPosition())
#     print("------------------------------------------------")
#     pass

# for y in [450, 1]:
#     CGA_y.MoveTo(y)
#     print("------------------------------------------------")
#     print(CGA_y.GetCurrentPosition())
#     print("------------------------------------------------")
#     pass

# for z in [225, 5]:
#     CGA_z.MoveTo(z)
#     print("------------------------------------------------")
#     print(CGA_z.GetCurrentPosition())
#     print("------------------------------------------------")
#     pass

# for r in [270, -180]: 
#     CGA_r.MoveFor(r)
#     print("------------------------------------------------")
#     print(CGA_z.GetCurrentPosition())
#     print("------------------------------------------------")
#     pass

# for r in [360]: 
#     CGA_r.MoveTo(r)
#     print("------------------------------------------------")
#     print(CGA_z.GetCurrentPosition())
#     print("------------------------------------------------")
#     pass


