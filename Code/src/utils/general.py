# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 15:08:35 2021

@author: SI042101
"""


import cv2
import numpy as np
import os
import shutil


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
       z_max: float = 250
       r_max: float = 360
       
       n_images: int = 50
       # output_dir: str = "C:\\Users\SI042101\ETH\Master_Thesis\PyData"
    


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

def ResizeWithAspectRatio(image, width = None, height = None, inter = cv2.INTER_AREA):
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
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation = inter)

    
    
    