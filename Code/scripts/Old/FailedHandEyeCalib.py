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

from scipy.spatial.transform import Rotation as Rot
from utils.general import OffsetParameters, PCLConfigs, StreamConfigs, loadIntrinsics, deprojectPoints
from utils.point_cloud_operations2 import PointCloud
# import open3d as o3d

# from pyqtgraph.Qt import QtCore, QtGui
# import pyqtgraph.opengl as gl

cv2.destroyAllWindows()
plt.close()



CHECKERBOARD = {}
CHECKERBOARD["size"] = ((5,3))
CHECKERBOARD["scale"] = 0.02

# cali_data_dir = "J:\GitHub\Recon\Data\CalibrationData/20221130_162627_calibration.json"

# calibration_folder = "C:/Users/SI042101/ETH/Master_Thesis/Data/PyData/20230320_172148"
calibration_folder = "C:/Users/SI042101/ETH/Master_Thesis/Data/PyData/20230111_145344/calibration"

###################################################################################################

# get robot hand data in base coordinates [mm]

pose_data = pd.read_csv(os.path.join(calibration_folder, "pose_info.csv"))


subset = pose_data.get(["x_read", "y_read", "z_read", "r_read"])
t_hand = tuple(x[:3]/1000 for x in subset.to_numpy())
r_hand = tuple(np.array([0,0,x[3]])*np.pi/180 for x in subset.to_numpy())

    

size = CHECKERBOARD["size"]
scale = CHECKERBOARD["scale"]
    
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = []



objp = np.zeros((1, size[0] * size[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)*scale

image_folder = os.path.join(calibration_folder, "img")
image_list = os.listdir(image_folder)
image_list.sort()

depth_folder = os.path.join(calibration_folder, "depth")
depth_list = os.listdir(depth_folder)
depth_list.sort()


for i, image in enumerate(image_list):
    
    print(f"{i+1}/{len(image_list)}")
    
    ref = cv2.imread(os.path.join(image_folder, image))
    gray = cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    # print(ret)
    if ret == True:
     
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), CRITERIA)
       
        objpoints.append(objp)
        imgpoints.append(corners2)
        
        # img = cv2.drawChessboardCorners(ref.copy(), size, corners2, ret)
        # cv2.imshow("",img)
        # cv2.waitKey(1000)
    
    else:
        print(ret)
      


K_d, K_c,_ = loadIntrinsics()
_, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(round(mean_error/len(objpoints),3)) )




R, t = cv2.calibrateHandEye(r_hand, t_hand, rvecs, tvecs, cv2.CALIB_HAND_EYE_HORAUD)

T = np.vstack([np.hstack([R,t]), [0,0,0,1]])


R_vis = Rot.from_matrix(R).as_euler("XYZ", degrees=True)
print(np.round(R_vis,1))
print(t)
    

depth = cv2.imread(os.path.join(depth_folder, depth_list[i]),-1).astype(np.float32)*0.001
pts = deprojectPoints(depth, K_d)
pts2 = deprojectPoints(depth, mtx)

pcl = o3d.geometry.PointCloud()
pcl2 = o3d.geometry.PointCloud()

pcl.points = o3d.utility.Vector3dVector(pts)
pcl2.points = o3d.utility.Vector3dVector(pts2)

pcl.paint_uniform_color([1,0,0])
pcl2.paint_uniform_color([1,1,0])

pcl.transform(T)

origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=np.array([0., 0., 0.]))
o3d.visualization.draw_geometries([origin,pcl, pcl2])





