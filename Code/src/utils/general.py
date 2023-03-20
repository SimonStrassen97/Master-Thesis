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


from dataclasses import dataclass


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
    
    
@dataclass
class ParameterConfigs(ConfigBase):
    vis: bool = False
    method: str = "ransac"
    matcher: str = "flann"
    feature: str = "sift"
    
    
@dataclass
class StreamConfigs(ConfigBase):
    c_hfov: int = 848
    c_vfov: int = 480
    c_fps: str = 30
    
    d_hfov: int = 848
    d_vfov: int = 480
    d_fps: str = 30
    
    
@dataclass
class AxisConfigs(ConfigBase):
    x_max: float = 650
    y_max: float = 475
    z_max: float = 125
    r_max: float = 360
    
    n_images: int = 50

    # output_dir: str = "C:\\Users\SI042101\ETH\Master_Thesis\PyData"
    
@dataclass
class PCLConfigs(ConfigBase):
    
    voxel_size: float = 0.05
    
    # m
    depth_thresh: float = 1
    
    # CleanUp
    # m
    border_x: tuple = (-0.0, 0.700)
    border_y: tuple = (-0.01, 0.600)
    border_z: tuple = (-0.02, 0.5)
    
    # filters
    hp_radius: float = 75
    angle_thresh: float = 95
    std_ratio: float = 1
    
    nb_points: int = 10
    outlier_radius: float = 0.01
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
    
    
@dataclass
class OffsetParameters(ConfigBase):
    # Offset parameters between Camera and RGA coordinates in Arm coordinates (in arm coords)

    # mm
    x_cam: float = 65
    y_cam: float = 25
    z_cam: float = 25
    
    
    # deg
    r_x_cam: float = 0
    r_y_cam: float = 49
    r_z_cam: float = 0
    
  
    # Offset parameters between RGA coordinates and World (WT) coordinates (in world coords)
    
    # mm
    x_arm: float = 83
    y_arm: float = 77.15
    z_arm: float = 341.75
    
    
    
    

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
    
    
    crop_offset = 8
    K = np.array([[c.get("fx"), 0, c.get("cx")-crop_offset],
                         [0 , c.get("fy"), c.get("cy")],
                         [0 , 0 , 1]
                         ])
    
   
    return K

def loadIntrinsics(file=None):
    
    if not file:
        file = "./intrinsics.pkl"
        
    with open(file, "rb") as f:
        intrinsics = pickle.load(f)
    
    K = _getIntrinsicMatrix(intrinsics)

    return K, intrinsics
  

