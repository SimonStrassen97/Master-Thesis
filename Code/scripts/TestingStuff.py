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
import pickle

# from scipy.spatial.transform import Rotation as Rot
from utils.general import StreamConfigs, PCLConfigs, StreamConfigs, OffsetParameters
from utils.general import loadIntrinsics 
from utils.point_cloud_operations2 import PointCloud
from utils.camera_operations import StereoCamera
from utils.worktable_operations import Object, Worktable
from scipy.spatial.transform import Rotation as Rot
# from utils.dpt_monodepth import run
# import open3d as o3d

# from pyqtgraph.Qt import QtCore, QtGui
# import pyqtgraph.opengl as gl

# cv2.destroyAllWindows()
plt.close()


###################################################################################################


cp = "/home/simonst/github/sparse-to-dense/results/wt.sparsifier=None.samples=0.modality=rgbd.arch=resnet18.decoder=deconv2.criterion=l1.lr=0.01.bs=4.pretrained=True/checkpoint-99.pth.tar"
path = "C:/Users/SI042101\ETH\Master_Thesis/Data/PyData/20230227_111004"


cam_offset = OffsetParameters(r_z_cam=-18, y_cam=22, x_cam=55)
pcl_configs = PCLConfigs(outliers=False, 
                         voxel_size=0.001, 
                         n_images=10,
                         hp_radius=200,
                         angle_thresh=90,
                         std_ratio=1,
                         # registration_method=None,
                         )

K_d, K_c, intr = loadIntrinsics()
dist = np.array(intr["color"]["dist"])


one = 0
two = 1

img_1 = cv2.imread(os.path.join(path, "img", str(one).zfill(4) + "_img.png"), -1)[:,:,::-1]
img_2 = cv2.imread(os.path.join(path, "img", str(one).zfill(4) + "_img.png"), -1)[:,:,::-1]

gray_1 = cv2.cvtColor(img_1,cv2.COLOR_RGB2GRAY)
gray_2 = cv2.cvtColor(img_2,cv2.COLOR_RGB2GRAY)

with open(os.path.join(path, "pose_info.csv"), "r") as f:
    pose_data = pd.read_csv(f).to_numpy()
    
X_1 = pose_data[one,5:]
X_2 = pose_data[two,5:]
D = X_2 - X_1


R = Rot.from_euler("ZXY", [D[3],0,0]).as_matrix()

R = np.array([[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]])
t = np.array([D[:3]]).T

R1, R2, P1, P2, Q, _, _  = cv2.stereoRectify(K_c, dist, K_c, dist, img_1.shape[1::-1], R, t)

map_1 = cv2.initUndistortRectifyMap(K_c, dist, R1, P1, img_1.shape[1::-1], cv2.CV_32FC1)
map_2 = cv2.initUndistortRectifyMap(K_c, dist, R2, P2, img_1.shape[1::-1], cv2.CV_32FC1)

rect_1 = cv2.remap(gray_1,map_1[0],map_1[1], cv2.INTER_LANCZOS4)
rect_2 = cv2.remap(gray_2,map_2[0],map_2[1], cv2.INTER_LANCZOS4)

stereo = cv2.StereoBM_create()
# stereo = cv2.StereoSGBM_create()
# stereo.setMode(cv2.STEREO_SGBM_MODE_HH)



def nothing(x):
    pass
 
cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)
 
cv2.createTrackbar('numDisparities','disp',1,17,nothing)
cv2.createTrackbar('blockSize','disp',5,50,nothing)
cv2.createTrackbar('preFilterType','disp',1,1,nothing)
cv2.createTrackbar('preFilterSize','disp',2,25,nothing)
cv2.createTrackbar('preFilterCap','disp',5,62,nothing)
cv2.createTrackbar('textureThreshold','disp',10,100,nothing)
cv2.createTrackbar('uniquenessRatio','disp',8,100,nothing)
cv2.createTrackbar('speckleRange','disp',10,100,nothing)
cv2.createTrackbar('speckleWindowSize','disp',3,25,nothing)
cv2.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
cv2.createTrackbar('minDisparity','disp',1,25,nothing)
# cv2.createTrackbar('P1','disp',5,100,nothing)
# cv2.createTrackbar('P2','disp',50,200,nothing)

while True:
 
   
    # Updating the parameters based on the trackbar positions
    numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
   
    blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
    preFilterType = cv2.getTrackbarPos('preFilterType','disp')
    preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5
    preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
    textureThreshold = cv2.getTrackbarPos('textureThreshold','disp')
    # P1 = cv2.getTrackbarPos('P1','disp')*10
    # P2 = cv2.getTrackbarPos('P2','disp')*10
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
    # stereo.setP1(P1)
    # stereo.setP2(P2)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)
 
    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(rect_1,rect_2)
    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it 
    # is essential to convert it to CV_32F and scale it down 16 times.
 
    # Converting to float32 
    disparity = disparity.astype(np.float32)
 
    # Scaling down the disparity values and normalizing them 
    disparity = (disparity/16.0 - minDisparity)/numDisparities
 
    # Displaying the disparity map
    cv2.imshow("disp",disparity)
 
    # Close window using esc key
    if cv2.waitKey(1) == 27:
      break
#################################################################33



# # get robot hand data in base coordinates [mm]
# pose_data = pd.read_csv(os.path.join(path, "pose_info.csv"))

# img_folder = os.path.join(path, "img")
# img_list = os.listdir(img_folder)

# for i, name in enumerate(img_list[:5]):
    
#     file = os.path.join(img_folder, name)
#     img = cv2.imread(file, -1)
#     gray_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray_, 5)
#     ret, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
#     thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
#     thresh = thresh[250:450,:]


    
#     h, w = gray.shape
    
    
#     startAngle = -25
#     stopAngle = -15
#     delta = 0.05
    
#     startAngle = startAngle*np.pi/180
#     stopAngle = stopAngle*np.pi/180
#     delta = delta*np.pi/180
    
#     steps = int((stopAngle-startAngle)/delta)
#     cache = np.zeros(int(steps))
#     check = 0
    
#     for i in range(steps):
        
    
#         y = w/2 * np.tan(startAngle-i*delta)
#         dim = (2*int(abs(y)+10), w)
#         offset = int(dim[0]/2)
#         corners = np.zeros((2,2), int)
        
#         corners[0] = [0, offset + y]
#         corners[1] = [w, offset - y]
    
        
#         blank = np.full(dim, 255, dtype = np.uint8)
#         stamp = cv2.line(blank, tuple(corners[0]),  tuple(corners[1]), 0, 5)
#         blurred_stamp = cv2.GaussianBlur(stamp, (3,3),0)
        
#         ## look at max value of cross correlation --> max value of max values is the best fit
#         # cross_correlation_up = cv2.matchTemplate(edges, upstamp,  method=cv2.TM_CCOEFF_NORMED)
#         cross_correlation = cv2.matchTemplate(thresh, blurred_stamp,  method=cv2.TM_CCOEFF_NORMED)
#         cache[i] = np.amax(cross_correlation)
    
    
#     x = np.arange(0,int(steps))
#     y  = cache
#     coefs = poly.polyfit(x, y, 15)
#     ffit = poly.Polynomial(coefs)
    
#     plt.plot(cache)
#     plt.plot(x, ffit(x))
    
#     max1 = np.amax(cache[0:int(steps/2)])
#     maxind1 = np.argmax(cache[0:int(steps/2)])
    
    
#     tilt_angle = (startAngle +  maxind1 * delta)*180/np.pi
#     print(tilt_angle)
    
    
    









