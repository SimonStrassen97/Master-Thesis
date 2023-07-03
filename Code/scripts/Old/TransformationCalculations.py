# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:59:41 2023

@author: SI042101
"""

import os
import json
import numpy as np 
from scipy.spatial.transform import Rotation as Rot






def getCam2Arm(data):
    
    x,y,z,r = data["read"]
    
    pin2cam = np.array([63.33,9,86.16
                        ])
    
    R = Rot.from_euler("z", r, degrees=True)
    pin2cam = R.apply(pin2cam)
    # print(pin2cam)
    
    
    arm2pin = np.array([x,
                        y,
                        -z])
             
    R_c2p = Rot.from_euler("ZYZ", (r, 90+50, -90), degrees=True).as_matrix()
    t_c2p = np.expand_dims(pin2cam, axis=1)
    
    T_c2p = np.hstack([R_c2p, t_c2p])
    T_c2p = np.vstack([T_c2p, np.array([0,0,0,1])])
    
    
    T_p2c = np.linalg.inv(T_c2p)
    
    
    R_p2a = np.eye(3)
    t_p2a = np.expand_dims(arm2pin, axis=1)
    
    T_p2a = np.hstack([R_p2a, t_p2a])
    T_p2a = np.vstack([T_p2a, np.array([0,0,0,1])])
    
    T_a2p = np.linalg.inv(T_p2a)
    
    T_c2a = T_p2a @ T_c2p
    T_a2c = T_p2c @ T_a2p 
    
    return T_c2a, T_a2c
    

def getWorld2Arm():
    

    with open("./WorldArm_calibration.txt", "r") as f:
        d = json.load(f)
        
        T_a2w = np.array(d["T_a2w"])
        T_w2a = np.array(d["T_w2a"])
    
    return T_a2w, T_w2a

def getWorld2Cam(data):
    
    
    T_c2a, T_a2c = getCam2Arm(data)
    T_a2w, T_w2a = getWorld2Arm()
    T_w2c = T_a2c @ T_w2a
    T_c2w = T_a2w @ T_c2a 
         
    return T_w2c, T_c2w


path = "C:/Users/SI042101/ETH/Master_Thesis/Data/PyData/20230417_174256"


data_folder = os.path.join(path, "data")
data_files = os.listdir(data_folder)
data_files.sort()


data_files = [os.path.join(data_folder, x) for x in data_files]

for d in data_files:
    
    with open(d, "r") as f:
        data = json.load(f)
    
    
    T_w2c, T_c2w = getWorld2Cam(data)
    r = data["read"][-1]
    c_w = T_c2w @ np.array([0,0,0,1])
    c_w = np.append(c_w[:-1], r)
    # print(c_w)
    
    data["cam"] = c_w.tolist()
    data["T_c2w"] = T_c2w.tolist()
    data["T_w2c"] = T_w2c.tolist()
    # del data["c_w"]
    
    json_obj = json.dumps(data, indent=4)
    
    # with open(d, "w") as f:
    #     f.write(json_obj)
    






