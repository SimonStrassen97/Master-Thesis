# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:00:20 2022

@author: SI042101
"""


import os, sys



import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d
import copy
import numpy.polynomial.polynomial as poly

from scipy.spatial.transform import Rotation as Rot
from utils.general import CameraOffset, PCLConfigs, StreamConfigs, ResizeWithAspectRatio
from utils.point_cloud_operations2 import PointCloud
# import open3d as o3d

# from pyqtgraph.Qt import QtCore, QtGui
# import pyqtgraph.opengl as gl

cv2.destroyAllWindows()
plt.close()

path = "C:/Users/SI042101/ETH/Master_Thesis/Data/PyData/20230110_165007"

###################################################################################################

# get robot hand data in base coordinates [mm]

pose_data = pd.read_csv(os.path.join(path, "pose_info.csv"))

img_folder = os.path.join(path, "img")
img_list = os.listdir(img_folder)

for i, name in enumerate(img_list[:5]):
    
    file = os.path.join(img_folder, name)
    img = cv2.imread(file, -1)
    gray_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray_, 5)
    ret, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    thresh = thresh[250:450,:]


    
    h, w = gray.shape
    
    
    startAngle = -25
    stopAngle = -15
    delta = 0.05
    
    startAngle = startAngle*np.pi/180
    stopAngle = stopAngle*np.pi/180
    delta = delta*np.pi/180
    
    steps = int((stopAngle-startAngle)/delta)
    cache = np.zeros(int(steps))
    check = 0
    
    for i in range(steps):
        
    
        y = w/2 * np.tan(startAngle-i*delta)
        dim = (2*int(abs(y)+10), w)
        offset = int(dim[0]/2)
        corners = np.zeros((2,2), int)
        
        corners[0] = [0, offset + y]
        corners[1] = [w, offset - y]
    
        
        blank = np.full(dim, 255, dtype = np.uint8)
        stamp = cv2.line(blank, tuple(corners[0]),  tuple(corners[1]), 0, 5)
        blurred_stamp = cv2.GaussianBlur(stamp, (3,3),0)
        
        ## look at max value of cross correlation --> max value of max values is the best fit
        # cross_correlation_up = cv2.matchTemplate(edges, upstamp,  method=cv2.TM_CCOEFF_NORMED)
        cross_correlation = cv2.matchTemplate(thresh, blurred_stamp,  method=cv2.TM_CCOEFF_NORMED)
        cache[i] = np.amax(cross_correlation)
    
    
    x = np.arange(0,int(steps))
    y  = cache
    coefs = poly.polyfit(x, y, 15)
    ffit = poly.Polynomial(coefs)
    
    plt.plot(cache)
    plt.plot(x, ffit(x))
    
    max1 = np.amax(cache[0:int(steps/2)])
    maxind1 = np.argmax(cache[0:int(steps/2)])
    
    
    tilt_angle = (startAngle +  maxind1 * delta)*180/np.pi
    print(tilt_angle)
    
    
    









