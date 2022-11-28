# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:57:57 2022

@author: SI042101
"""

import os
import glob
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime



# edit JSONEncoder to convert np arrays to list
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

image_folder = os.path.realpath("C:/Users/SI042101/ETH/Master_Thesis/Images/Calibration_Test/")
image_list = glob.glob(os.path.join(image_folder,"*"))



CHECKERBOARD = (10,7)
SCALE = 25
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
SAVE_DATA = True

objpoints = []
imgpoints = []

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)*SCALE


for image in image_list:
    ref = cv2.imread(os.path.join(image_folder, image))
    gray = cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray,CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret == True:
     
       # refining pixel coordinates for given 2d points.
       
       corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), CRITERIA)
       
       objpoints.append(objp)
       imgpoints.append(corners2)
        
       img = cv2.drawChessboardCorners(ref.copy(), CHECKERBOARD, corners2, ret)
      


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
 

camera = {}


for variable in ['ret', 'mtx', 'dist', 'rvecs', 'tvecs']:
    camera[variable] = eval(variable)

if SAVE_DATA:
    time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"../../../Data/CalibrationData/{time_string}_calibration.json"
    with open(output_dir, 'w') as f:
        json.dump(camera, f, indent=4, cls=NumpyEncoder)
    
        
    
    
    
    
    
    
    