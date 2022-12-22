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
from mpl_toolkits import mplot3d


# from PyTeMotion.AxisFunctions import Axis
from utils.camera_operations import Camera, StereoCamera, PseudoStereoCamera
from utils.general import ResizeWithAspectRatio
import utils.dpt_monodepth as dpt
from utils.general import StreamConfigs


import open3d as o3d

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

cv2.destroyAllWindows()
plt.close()



CHECKERBOARD = {}
CHECKERBOARD["size"] = (10,7)
CHECKERBOARD["scale"] = 25

cali_data_dir = "J:\GitHub\Recon\Data\CalibrationData/20221130_162627_calibration.json"

calibration_folder = os.path.realpath("C:/Users/SI042101/ETH/Master_Thesis/Images/Calibration_Test/")
input_folder = os.path.realpath("C:/Users/SI042101/ETH/Master_Thesis/Images/Images MT/Full WT/")


# for monocular ml approach
output_folder = os.path.realpath("C:/Users/SI042101/ETH/Master_Thesis/Images/Images MT/Full WT Out/")
weights_path = "J:/GitHub/DPT/weights/dpt_hybrid-midas-501f0c75.pt"
# model_type = "dpt_hybrid"
model_type = "dpt_hybrid_nyu"
optimize = True
absolute_depth = True


###################################################################################################

configs = StreamConfigs
cam = StereoCamera(activate_adv=False)
cam.startStreaming(configs)
cam.getFrame()
# if cali_data_dir:
#     cam.load_cam_params(cali_data_dir)
# else:
#     cam.calibrate(calibration_folder, CHECKERBOARD, save_data=True)
    

for name in os.listdir(input_folder):
    
    if not "2324" in name:
        continue
    img = cv2.imread(os.path.join(input_folder, name))
    # depth = cv2.imread(os.path.join(output_folder, f"{orig[:-4]}.png"))
    # rect_img = cam.undistort(img)
    
    inv_depth = dpt.run(img, name, output_folder, weights_path, model_type=model_type, absolute_depth=absolute_depth)
    # depth = inv_depth.max()+inv_depth.min()-inv_depth
    
    plt.imshow(inv_depth)
    
    
# app = QtGui.QGuiApplication([])
# gl_widget = gl.GLViewWidget()
# gl_widget.show()
# gl_grid = gl.GLGridItem()
# gl_widget.addItem(gl_grid)


    

# cam.Depth2PCL(img, inv_depth, cam.camera_params["mtx"])
# T = np.eye(4)
# T[:3, :3] = pcl.pcl.get_rotation_matrix_from_xyz((np.pi, 0, np.pi/3*2))
# T[0, 3] = 5000
# T[1, 3] = 0
# T[2, 3] = 5000
# pcl.CamToWorld(-T)
cam.visualizePCL()

                            
    
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(pcl[0], pcl[1], pcl[2])
    
   



  

# fig1, ax1 = plt.subplots(1,2)
# ax1[0].imshow(img)
# ax1[1].imshow(rect_img)


####################################################################################################

# stereo_cam = PseudoStereoCamera()

# stereo_cam.load_cam_params(cali_data_dir)

# img_l = cv2.imread(os.path.join(f"{input_folder}/left.png"),0)
# # rect_img_l = cam.undistort(img_l, mode=0)

# img_r= cv2.imread(os.path.join(f"{input_folder}/right.png"),0)
# # rect_img_r = cam.undistort(img_r, mode=0)

# stereo_cam.calibrate(input_folder, CHECKERBOARD)

# rect_img_r, rect_img_l = stereo_cam.stereoUndistort(img_l, img_r)

# stereo = cv2.StereoSGBM_create(numDisparities=64, blockSize=9)

# disparity = stereo.compute(rect_img_l, rect_img_r)
# # disparity = np.int16(disparity)

# fig2, ax2 = plt.subplots(1,3)
# ax2[0].imshow(rect_img_l)
# ax2[1].imshow(rect_img_r)
# ax2[2].imshow(disparity, "gray")






# ############################################


# cam2 = CameraMono()
# rel_depth = cv2.imread(test_dir,0).astype(np.uint8)
# cam2.load_cam_params(cali_data_dir)
# corrected_rel_depth = cam2.undistort(rel_depth)


