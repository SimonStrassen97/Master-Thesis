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

from scipy.spatial.transform import Rotation as Rot
from utils.general import StreamConfigs, ResizeWithAspectRatio
from utils.general import OffsetParameters, PCLConfigs, StreamConfigs
from utils.point_cloud_operations2 import PointCloud
from utils.camera_operations import StereoCamera
from utils.dpt_monodepth import run
# import open3d as o3d

# from pyqtgraph.Qt import QtCore, QtGui
# import pyqtgraph.opengl as gl

cv2.destroyAllWindows()
plt.close()


###################################################################################################

        
def getIntrinsicMatrix(intrinsics):
    
    c = intrinsics.get("color")
    d = intrinsics.get("depth")
    
    K_c = np.array([[c.get("fx"), 0, c.get("cx")],
                         [0 , c.get("fy"), c.get("cy")],
                         [0 , 0 , 1]
                         ])
    
    K_d = np.array([[d.get("fx"), 0, d.get("cx")],
                         [0 , d.get("fy"), d.get("cy")],
                         [0 , 0 , 1]
                         ])
   
    return K_c, K_d
    
def loadIntrinsics(file=None):
    
    if not file:
        file = "./intrinsics.pkl"
        
    with open(file, "rb") as f:
        intrinsics = pickle.load(f)
        
    K, _ = getIntrinsicMatrix(intrinsics)
    
    return K


root = "J:/GitHub/Datasets/Simple_WT"
WEIGHTS = "J://GitHub\MT_Temp/DPT/weights/dpt_hybrid-midas-501f0c75.pt"
out = os.path.join(root, "DPT_Out")

pd = os.path.join(root, "depth")
pi = os.path.join(root, "img")

img = cv2.imread(os.path.join(pi, "0000_img.png"), -1)
dmap = cv2.imread(os.path.join(pd, "0000_depth.png"), -1)
pp = "C://Users\SI042101\ETH\Master_Thesis\Data\PyData/20230110_165007\PCL/0000_pcl.ply"

dpt = run(img, "0000_img.png", output_path=out, model_path=WEIGHTS ,model_type="dpt_hybrid_nyu")

a = dmap.copy()
a[a==0] = dpt[a==0]
plt.imshow(a)

dpt/dmap



depth = dmap.astype(np.float64) * 0.0001
K = loadIntrinsics()


cam_offset = OffsetParameters(y_cam=25)

pcl_configs = PCLConfigs(voxel_size=0.005,
                          depth_thresh=1,
                          vis=False,
                          # color="gray",
                          n_images=8,
                          outliers=False,
                          hp_radius=75,
                          angle_thresh=95,
                          std_ratio=1,
                          nb_points=10,
                          outlier_radius=0.01,
                          recon_method="poisson",
                          registration_method="",
                          registration_radius=0.003,
                          coord_scale=0.1
                          )

pcl1 = PointCloud(pcl_configs,cam_offset)

pcl2 = PointCloud(pcl_configs,cam_offset)


pcl = pcl1.pcl_from_depth(img, depth, K)
pcl_ = o3d.io.read_point_cloud(pp)
R  = pcl_.get_rotation_matrix_from_xyz((np.pi, 0, 0))
pcl_.rotate(R, center=(0,0,0))

pcl.translate((0.5,0,0))

points_ = np.array(pcl_.points)
condition = points_[:,2] < 0.7
ind = np.where(condition)[0]
pcl_ = pcl_.select_by_index(ind)

points = np.array(pcl.points)
condition = points[:,2] < 0.7
ind = np.where(condition)[0]
pcl = pcl.select_by_index(ind)




o3d.visualization.draw_geometries([pcl, pcl_])











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
    
    
    









