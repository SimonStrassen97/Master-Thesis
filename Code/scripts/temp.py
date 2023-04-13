#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:39:00 2023

@author: simonst
"""
    
    
import time
import os

import numpy as np
import open3d as o3d
import copy
import cv2
import matplotlib.pyplot as plt
import scipy.optimize
import skimage.exposure
from numpy.random import default_rng


from utils.general import StreamConfigs, PCLConfigs, StreamConfigs, OffsetParameters
from utils.general import loadIntrinsics 
from utils.camera_operations import StereoCamera
from utils.worktable_operations import Object, Worktable



def d2s_edge(rgb, depth, max_depth=2**16, num_samples=325632, dilate_kernel=3, dilate_iterations=1):
    
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

    depth_mask = np.bitwise_and(depth != 0.0, depth <= max_depth)

    edge_fraction = float(num_samples) / np.size(depth)

    mag = cv2.magnitude(gx, gy)
    min_mag = np.percentile(mag[depth_mask], 100 * (1.0 - edge_fraction))
    mag_mask = mag >= min_mag

    if dilate_iterations >= 0:
        kernel = np.ones((dilate_kernel, dilate_kernel), dtype=np.uint8)
        cv2.dilate(mag_mask.astype(np.uint8), kernel, iterations=dilate_iterations)

    mask = np.bitwise_and(mag_mask, depth_mask)
    
    dep_sp = depth * mask.reshape(depth.shape).astype(np.uint16)
    
    return dep_sp

def d2s_random(depth, percentage=0.5):
    height, width = depth.shape

    idx_nnz = np.flatnonzero(depth)
    num_samples = int(depth.size * percentage)

    num_idx = len(idx_nnz)
    idx_sample = np.random.permutation(num_idx)[:num_samples]

    idx_nnz = idx_nnz[idx_sample[:]]

    mask = np.zeros((height*width))
    mask[idx_nnz] = 1.0
    
    dep_sp = depth * mask.reshape(depth.shape).astype(np.uint16)

    return dep_sp

def d2s_spots(depth, thresh=175):
    
    # seedval = 55
    # rng = default_rng(seed=seedval)

    # define image size
    height, width = depth.shape
    
    # create random noise image
    noise = np.random.randint(0, 255, (height,width), np.uint8)
    
    # blur the noise image to control the size
    blur = cv2.GaussianBlur(noise, (0,0), sigmaX=15, sigmaY=15, borderType = cv2.BORDER_DEFAULT)
    
    # stretch the blurred image to full dynamic range
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)
    
    # threshold stretched image to control the size
    thresh = cv2.threshold(stretch, thresh, 255, cv2.THRESH_BINARY)[1]
    
    # apply morphology open and close to smooth out shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    mask = (255 - result) /255
      
    dep_sp = depth * mask.reshape(depth.shape).astype(np.uint16)
    
    return dep_sp
    
    
    

path = "/home/simonst/github/Datasets/wt/raw/20230227_181401/"

d = cv2.imread(os.path.join(path, "depth", "0001_depth.png"), -1)
img = cv2.imread(os.path.join(path, "img", "0001_img.png"), -1)


# fig, ax = plt.subplots(1,2)
# ax[0].imshow(d)
# ax[1].imshow(img)

sparse = d2s_random(d)
sparse_edge = d2s_edge(img, d)
sparse_dots = d2s_spots(d)

fig, ax = plt.subplots(1,4)
ax[0].imshow(d)
ax[2].imshow(sparse)
ax[1].imshow(sparse_edge)
ax[3].imshow(sparse_dots)








