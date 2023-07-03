# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 15:08:35 2021

@author: SI042101
"""


import cv2
import numpy as np
import os
import shutil
import pickle
import json
import torch
import argparse

from dataclasses import dataclass
from CoordConv import AddCoordsNp
from dataloaders.transforms import ToTensor

@dataclass
class ConfigBase:

    def verify(self):
        pass

    def to_dict(self):
        ret = dict()
        for item in dir(self):
            val = getattr(self, item)
            if item.startswith("__") or callable(val):
                continue
            if isinstance(val, ConfigBase):
                ret[item] = val.to_dict()
            else:
                ret[item] = val
        return ret
    
    
# @dataclass
# class ParameterConfigs(ConfigBase):
#     vis: bool = False
#     method: str = "ransac"
#     matcher: str = "flann"
#     feature: str = "sift"
    
    
@dataclass
class StreamConfigs(ConfigBase):
    c_hfov: int = 848
    c_vfov: int = 480
    c_fps: str = 30
    
    d_hfov: int = 848
    d_vfov: int = 480
    d_fps: str = 30
    
    
    # c_hfov: int = 1280
    # c_vfov: int = 720
    # c_fps: str = 30
    
    # d_hfov: int = 1280
    # d_vfov: int = 720
    # d_fps: str = 30
    
    
    
    
@dataclass
class AxisConfigs(ConfigBase):
    
    x_max: float = 650
    y_max: float = 475
    z_max: float = 125
    r_max: float = 360


    # output_dir: str = "C:\\Users\SI042101\ETH\Master_Thesis\PyData"
    
@dataclass
class PCLConfigs(ConfigBase):
    
    
    verbose: bool = False
    
    pre_voxel_size: float = 0.0
    voxel_size: float = 0.0
    
    # m
    depth_thresh: float = 1
    
    # CleanUp
    # m
    border_x: tuple = (-0.01, 0.800)
    border_y: tuple = (-0.01, 0.600)
    border_z: tuple = (-0.01, 0.1575)
    # border_z: tuple = (-0.01, 0.175)
    
    #########################################
    # filters
    filters: bool = True
    
    #hidden points
    hp_radius: float = 75
    angle_thresh: float = 95
    
    # statistical outlier 
    nb_points_stat: int = 50
    std_ratio_stat: float = 1
    
    # box filter
    nb_points_box: int = 10
    box_radius: float = 0.01
    ###########################################
    
    
    # mesh
    recon_method: str = "poisson"
    
    # registration
    registration_method: str = "plane"
    registration_radius: float = 0.003
    
    # registration_method: str = "color"
    # registration_radius: float = 0.05
    
    
    # vis
    vis: bool = True
    coord_frame: bool = True
    coord_scale: float = 0.1
    outliers: bool = True
    color: str = None
    
    n_images: int = 8
    
    # resize: bool = False
    
    
    
@dataclass
class OffsetParameters(ConfigBase):
    # Offset parameters between Camera and RGA coordinates in Arm coordinates (in arm coords)

    # mm
    x_cam: float = 65
    y_cam: float = 25
    z_cam: float = 25
    
    
    # deg
    r_x_cam: float = 0
    r_y_cam: float = 50
    r_z_cam: float = 0
  
    # Offset parameters between RGA coordinates and World (WT) coordinates (in world coords)
    
    # mm
    x_arm: float = 83
    y_arm: float = 77.15
    z_arm: float = 341.75
    
    
@dataclass
class CalibParams(ConfigBase):
    # Offset parameters between Camera and reference pin's tip in world coordinates (

    # mm
    x_pin2cam: float = 63.33
    y_pin2cam: float = 9
    z_pin2cam: float = 86.16
    
    # deg
    r_x_pin2cam: float = 0
    r_y_pin2cam: float = 50
    r_z_pin2cam: float = 0
    
    x_world2ref: float = 83.0
    y_world2ref: float = 325.0
    z_world2ref: float = 0
    
    r_x_world2ref: float = 0.0
    r_y_world2ref: float = 0.0
    r_z_world2_ref: float = 0.0
        
    

def detectBlurryImgs(path, thresh=30, delete=True):
    imgs = os.listdir(path)
    
    for name in imgs:
        file = os.path.join(path,imgs)
        img = cv2.imread(file, 0)
        
        lapl = cv2.Laplacian(img, cv2.CV_64F)
        score = np.var(lapl)
        print(f"Score is {score}.")
        
        if score < thresh:
            print("Deleting img {name}")
            os.remove(file)
            
            
def _detectBlurryImgs(img, thresh=30, delete=True):
    
    blurry = False
    lapl = cv2.Laplacian(img, cv2.CV_64F)
    score = np.var(lapl)
    print(f"Blurry score is {score}.")
    
    if score < thresh:
        blurry = True
    
    return blurry
        
        
        
    
def createSnapshots(path, every_x_sec):
    # Read the video from specified path
    vid = cv2.VideoCapture(path)
    head, tail = os.path.split(path)
    output_folder = os.path.join(head, tail[:-4])
    
    fps = round(vid.get(cv2.CAP_PROP_FPS))
    print(fps)
    
    try:
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)
    
    except:
        os.makedirs(output_folder)
        
    # frame
    currentframe = 0
    
    while (True):
       
        # reading from frame
        ret, frame = vid.read()
    
        if ret:
            
            if (currentframe / fps) % every_x_sec == 0:
                
                blurry = _detectBlurryImgs(frame)
                
                if blurry:
                    print(f"Skipping blurry img...")
                    continue
               
                # if video is still left continue creating images
                name = os.path.join(output_folder, f"frame_{currentframe}.jpg")
                print('Creating...' + name)
        
                # writing the extracted images
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                cv2.imwrite(name, frame)
        
                # increasing counter so that it will
                # show how many frames are created
                
            currentframe += 1
        else:
            break
    
    # Release all space and windows once done
    vid.release()
    cv2.destroyAllWindows()
    
    return output_folder


def deprojectPoints(dmap, K, remove_zeros=True):

        
    R, C = np.indices(dmap.shape)
    fx = K[0,0]
    fy = K[1,1]
    cy = K[0,2]
    cx = K[1,2]

    R = np.subtract(R, cx)
    R = np.multiply(R, dmap)
    R = np.divide(R, fx)

    C = np.subtract(C, cy)
    C = np.multiply(C, dmap)
    C = np.divide(C, fy)
    
    # pts = np.column_stack((dmap.ravel()/scale, R.ravel(), -C.ravel()))
    pts = np.column_stack((C.ravel(), R.ravel(), dmap.ravel()))
    if remove_zeros:
        pts = pts[pts[:,2]!=0]

    return pts

def projectPoints(pts, K, out_size):
    
    depth = np.zeros(out_size)
    x,y,z = np.array_split(pts, 3, axis=1)
    fx = K[0,0]
    fy = K[1,1]
    cy = K[0,2]
    cx = K[1,2]
    
    R = np.multiply(y, fx)
    R = np.divide(R, z)
    R = np.add(R, cx)
    
    C = np.multiply(x, fy)
    C = np.divide(C, z)
    C = np.add(C, cy)
    
    R = np.round(R, 4)
    C = np.round(C, 4)
    
    
    r_mask = np.array([val.is_integer() for val in R.squeeze()])
    c_mask = np.array([val.is_integer() for val in C.squeeze()])
    mask = r_mask&c_mask
    

    R = R[mask].astype(int)
    C = C[mask].astype(int)

    z = z[mask]
    
    depth[R,C] = z
    
    return depth


def ResizeViaProjection(depth, K, out_size):

    
    pts = deprojectPoints(depth, K)
    
    size = depth.shape
    ratio = (out_size[0]/size[0], out_size[1]/size[1])
    
    K_new = K.copy()
    K_new[0] *= ratio[0]
    K_new[1] *= ratio[1]
    
    depth = projectPoints(pts, K_new, out_size)
    
    return depth, K_new
    
    
    


def ResizeWithAspectRatio(image, width = None, height = None, inter = cv2.INTER_NEAREST):
    """
    Selecting either a width or hight automatically adjusts the other variable to maintain
    the aspect ratio.
    
    Parameters
    ----------
    image : Array of uint8.
    width : int, The default is None.
    height : int, The default is None.
    inter : The default is cv2.INTER_AREA.

    Returns
    -------
    Array of uint8, resized image

    """
    dim = None
    r = 1 
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation = inter), r

def Crop(img, output_size):
    """Get parameters for ``crop`` for center crop.

    Args:
        img (numpy.ndarray (C x H x W)): Image to be cropped.
        output_size (tuple): Expected output size of the crop.

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for center crop.
    """
    h = img.shape[0]
    w = img.shape[1]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))

    if img.ndim == 3:
        return img[i:i+th, j:j+tw, :]
    elif img.ndim == 2:
        return img[i:i + th, j:j + tw]
 

def _getIntrinsicMatrix(intrinsics):
    
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

def loadIntrinsics(path=None):
    

    if not path: 
        path ="."
    try:
        
    
        file = os.path.join(path, "intrinsics.txt")
            
        with open(file, "r") as f:
            intrinsics = json.load(f)
            
    except:
        
        file = os.path.join(path, "intrinsics.pkl")
            
        with open(file, "rb") as f:
            intrinsics = pickle.load(f)
    
    
    K_c, K_d = _getIntrinsicMatrix(intrinsics)


    return K_d, K_c, intrinsics
  
def prepare_s2d_input(img, depth, K):
     
     crop_size = (224, 416)
     # crop_size = (480, 832)
     img, ratio = ResizeWithAspectRatio(img, height=240)
     depth_, _ = ResizeWithAspectRatio(depth, height=240)
     depth, K_new = ResizeViaProjection(depth, K, out_size=(240,424))
   
     # K_new = K.copy()
     cur_size = img.shape
     diff = (cur_size[0]-crop_size[0], cur_size[1]-crop_size[1])
     K_new[0,2] -= diff[1]/2
     K_new[1,2] -= diff[0]/2
     
     img = Crop(img, crop_size)
     depth = Crop(depth, crop_size)
    
     rgb = np.asfarray(img, dtype='float32') / 255
     depth = np.asfarray(depth, dtype="float32")
     depth = np.expand_dims(depth, -1)        

     position = AddCoordsNp(224, 416)
     # position = AddCoordsNp(480, 832)
     position = position.call()
     
     candidates = {"rgb": rgb, "d": depth, "gt": depth, \
                   'position': position, 'K': K_new}
     
     to_tensor = ToTensor()
     to_float_tensor = lambda x: to_tensor(x).float()

     items = {
         key: to_float_tensor(val).unsqueeze(0)
         for key, val in candidates.items() if val is not None
     }
  
     return items
    

def ME(x,y):
    
    e = x-y
    me = e.mean()
    
    return me

def MAE(x,y):
    
    ae = abs(x-y)
    mae = ae.mean()
    
    return mae


def MSE(x,y):
    
    se = (x-y)**2
    mse = se.mean()
    
    return mse


def RMSE(x,y):
    
    se = (x-y)**2
    mse = se.mean()
    rmse = np.sqrt(mse)
    
    return rmse                                                       


# parser = argparse.ArgumentParser(description='Sparse-to-Dense')
# parser.add_argument('-n',
#                     '--network-model',
#                     type=str,
#                     default="pe",
#                     choices=["e", "pe"],
#                     help='choose a model: enet or penet'
#                     )
# parser.add_argument('--cpu',
#                     type=bool,
#                     default=True
#                     )
# #geometric encoding
# parser.add_argument('-co', '--convolutional-layer-encoding', default="xyz", type=str,
#                     choices=["std", "z", "uv", "xyz"],
#                     help='information concatenated in encoder convolutional layers')

# #dilated rate of DA-CSPN++
# parser.add_argument('-d', '--dilation-rate', default="2", type=int,
#                     choices=[1, 2, 4],
#                     help='CSPN++ dilation rate')

# args = parser.parse_args()

