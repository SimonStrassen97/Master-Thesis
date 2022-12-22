# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 17:03:21 2022

@author: SI042101
"""


import cv2
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils.camera_operations import Camera
from utils.general import ResizeWithAspectRatio

cv2.destroyAllWindows()

cali_data_dir = "J:\GitHub\Recon\Data\CalibrationData/20221130_162627_calibration.json"

calibration_folder = os.path.realpath("C:/Users/SI042101/ETH/Master_Thesis/Images/Calibration_Test/")
input_folder = os.path.realpath("C:/Users/SI042101/ETH/Master_Thesis/Images/Stereo_Test/")


CHECKERBOARD = {}
CHECKERBOARD["size"] = (10,7)
CHECKERBOARD["scale"] = 25
cam = Camera()

if cali_data_dir:
    cam.load_cam_params(cali_data_dir)
else:
    cam.calibrate(calibration_folder, CHECKERBOARD, save_data=True)
    

img_l = cv2.imread(os.path.join(f"{input_folder}/left2.png"),0)
rect_img_l = cam.undistort(img_l, mode=0)

img_r= cv2.imread(os.path.join(f"{input_folder}/right2.png"),0)
rect_img_r = cam.undistort(img_r, mode=0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=7)
disparity = stereo.compute(rect_img_l, rect_img_r)

fig, ax = plt.subplots(1,3)
ax[0].imshow(rect_img_l)
ax[1].imshow(rect_img_r)
ax[2].imshow(disparity, "gray")


def nothing(x):
    pass

cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',1900,1200)
 
cv2.createTrackbar('numDisparities','disp',20,30,nothing)
cv2.createTrackbar('blockSize','disp',5,50,nothing)
cv2.createTrackbar('preFilterType','disp',1,1,nothing)
cv2.createTrackbar('preFilterSize','disp',5,25,nothing)
cv2.createTrackbar('preFilterCap','disp',30,62,nothing)
cv2.createTrackbar('textureThreshold','disp',10,100,nothing)
cv2.createTrackbar('uniquenessRatio','disp',15,100,nothing)
cv2.createTrackbar('speckleRange','disp',0,100,nothing)
cv2.createTrackbar('speckleWindowSize','disp',0,25,nothing)
cv2.createTrackbar('disp12MaxDiff','disp',0,25,nothing)
cv2.createTrackbar('minDisparity','disp',0,25,nothing)
 
# Creating an object of StereoBM algorithm
stereo = cv2.StereoBM_create()
plt.close()
 
while True:
 

    # Updating the parameters based on the trackbar positions
    numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
    blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
    preFilterType = cv2.getTrackbarPos('preFilterType','disp')
    preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5
    preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
    textureThreshold = cv2.getTrackbarPos('textureThreshold','disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
    speckleRange = cv2.getTrackbarPos('speckleRange','disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
    minDisparity = cv2.getTrackbarPos('minDisparity','disp')
     
    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(preFilterType)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)
 
    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(rect_img_l,rect_img_r)
    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it 
    # is essential to convert it to CV_32F and scale it down 16 times.
 
    # Converting to float32 
    disparity = disparity.astype(np.float32)
 
    # Scaling down the disparity values and normalizing them 
    disparity = (disparity/16.0 - minDisparity)/numDisparities
 
    # Displaying the disparity map
    # resize = ResizeWithAspectRatio(disparity, width=900)
    rot = cv2.rotate(disparity, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imshow("disp", rot)
 
    # Close window using esc key
    key = cv2.waitKey(1)
    if key == 27:
      cv2.destroyAllWindows()
      break