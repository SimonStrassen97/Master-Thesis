# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:00:20 2022

@author: SI042101
"""


import os, sys


import cv2
import numpy as np
import matplotlib.pyplot as plt

import open3d as o3d
import copy
import numpy.polynomial.polynomial as poly
import pickle

# from scipy.spatial.transform import Rotation as Rot
from utils.general import StreamConfigs, PCLConfigs, StreamConfigs, OffsetParameters
from utils.general import loadIntrinsics, ResizeViaProjection, ResizeWithAspectRatio
from utils.general import deprojectPoints, projectPoints
from utils.point_cloud_operations2 import PointCloud
from utils.camera_operations import StereoCamera
from utils.worktable_operations import Object, Worktable
# from utils.dpt_monodepth import run
# import open3d as o3d

# from pyqtgraph.Qt import QtCore, QtGui
# import pyqtgraph.opengl as gl

# cv2.destroyAllWindows()
plt.close()


###################################################################################################

    

cp = "/home/simonst/github/sparse-to-dense/results/wt.sparsifier=None.samples=0.modality=rgbd.arch=resnet18.decoder=deconv2.criterion=l1.lr=0.01.bs=4.pretrained=True/checkpoint-99.pth.tar"
path = "/home/simonst/github/Datasets/wt/raw/20230227_174221"


cam_offset = OffsetParameters(r_z_cam=-18, y_cam=22, x_cam=55)
pcl_configs = PCLConfigs(outliers=False, 
                         voxel_size=0.001, 
                         n_images=10,
                         hp_radius=200,
                         angle_thresh=0,
                         std_ratio=1,
                         # registration_method=None,
                         )

K,_ = loadIntrinsics()

# depth = cv2.imread(os.path.join(path, "depth", "0001_depth.png"), -1)
# out_size = (240,424)
# depth_, K_ = ResizeViaProjection(depth, K, out_size)

# inter = cv2.resize(depth, out_size[::-1], cv2.INTER_NEAREST)


# fig, ax = plt.subplots(1,2)
# ax[0].imshow(inter)
# ax[1].imshow(depth_)



pcl = PointCloud(pcl_configs, cam_offset)
# pcl2 = PointCloud(pcl_configs, cam_offset)


pcl.load_PCL_from_depth(path, K) 
# pcl2.load_PCL_from_depth(path,K,run_s2d=cp)


pcl.ProcessPCL()
# pcl2.ProcessPCL()

def custom_draw(pcd):
    
    def rotate(vis):
        ctr = vis.get_view_control()
        ctr.rotate(1,0)
        return False
    
    o3d.visualization.draw_geometries_with_animation_callback([pcd], rotate)

custom_draw(pcl.unified_pcl)

# p = pcl.unified_pcl
# pts = np.asarray(p.points)



         
# gridified wt
   
# wt = Worktable()
# wt.gridify_wt(pts)
# wt.visualize()

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
    
    
    









