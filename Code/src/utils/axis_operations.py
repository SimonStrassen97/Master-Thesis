# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:01:02 2022

@author: SI042101
"""

import os
import numpy as np
import pandas as pd
import json
import cv2
import random
import time
from scipy.spatial.transform import Rotation as Rot


class AxisMover():
    def __init__(self, x, y, z, r, configs):
        # Axes
        self.CGA_x = x
        self.CGA_y = y
        self.CGA_z = z
        self.CGA_r = r
        
        self.checkpoints = []
        self.step_size = None
        
        self.configs = configs
        self.calib_params = None
        
        # ouput data
        self.data = []
        self.data_ = {}
        
        # Axis state
        self.move_counter = 0
        self.calibrated = False
        
        self._updatePos()
        
        
    def _verifyTargetPos(self, target):
        
        border = [self.configs.x_max, self.configs.y_max, self.configs.z_max, self.configs.r_max]
        valid = True
        
        for t, b in zip(target[:-1], border[:-1]):
            if t > b+0.1 or t < -0.1:
                valid = False
                print("Target position out of bounds.")
        
        return valid
                
            
            
    def _updatePos(self):
        
        self.WakeUp()
        self.current_pos =  np.round(np.array([self.CGA_x.GetCurrentPosition(),
                             self.CGA_y.GetCurrentPosition(),
                             -self.CGA_z.GetCurrentPosition(),
                             self.CGA_r.GetCurrentPosition()]),3)
        
        if self.calibrated:
            
            self.current_pos[3] -= self.rot_offset
        
        
    def getPos(self):
        
        self._updatePos()
        
        return self.current_pos
        
        
    
    def Sleep(self):
        
        self.CGA_x.TurnOff()
        self.CGA_y.TurnOff()
        self.CGA_r.TurnOff()
        self.CGA_z.ZeroG()
    
    def WakeUp(self):
        
        self.CGA_x.TurnOn()
        self.CGA_y.TurnOn()
        self.CGA_r.TurnOn()
        self.CGA_z.TurnOn()
        
        
        
    def MoveTo(self, dest):
        
        self.WakeUp()
        
        if len(dest)==3:
            (x,y,z) = dest
            r = None
        
        else:
            (x,y,z,r) = dest
            self._updatePos()
        
        valid = self._verifyTargetPos((x,y,z,r))
        
        if not valid:
            return False
        
        self.CGA_z.MoveTo(abs(z))
        self.CGA_x.MoveTo(x)
        self.CGA_y.MoveTo(y)
        
        if isinstance(r,(float,int)): 
            r_0 = self.current_pos[3]
            
            if round(r_0,1) > 180:
                r_0 -= 360
            if round(r,1) > 180:
                r -= 360
            
            self.rr = (r_0, r)
            
            if round(r_0,1) != round(r,1):
    
                self.CGA_r.MoveFor(-r_0)
                self.CGA_r.MoveFor(r)
        
        self._updatePos()
        
        return True
            
            
        
    def MoveTo_World(self, dest):
        
        if not self.calibrated and not self.T_w2a:
            print("Calibrate first.")
            return None
            
        
        if len(dest)==3:
            (x,y,z) = dest
            r = None
        
        else:
            (x,y,z,r) = dest
            
        p_w = np.array([x,y,z,1])
        # p_w[2] =  -p_w[2]
        
        p_a = self.T_w2a @ p_w
        p_a = p_a[:-1]
        
        if isinstance(r,(float,int)):
            p_a = np.append(p_a, r)
        
        self.MoveTo(p_a)
        
        
        
    def MoveFor(self, dist):
        
        self.WakeUp()
        
        if len(dist) == 3:
            (x,y,z) = dist
            r = 0
        else:
            (x,y,z,r) = dist
            
            
        pos = self.getPos()
        
        target = pos + dist
        
        valid = self._verifyTargetPos(target)
        
        if not valid:
            return False
        
        self.CGA_z.MoveFor(z)
        self.CGA_x.MoveFor(x)
        self.CGA_y.MoveFor(y)
        self.CGA_r.MoveFor(r)
        
        return True
   

    def _calcTotalDist(self, checkpoints):
        
        total_dist = 0
        prev_cp = None
        sections = []
        for i, cp in enumerate(checkpoints[:,0:3]):
            if i!=0:
                section = np.linalg.norm(cp-prev_cp)
                total_dist += section
                sections.append(section)
            prev_cp = cp
            
        return total_dist, sections
            
        
        
    def evalCPs(self, checkpoints, n_imgs):
        
        
        MAX_VALUES = [self.configs.x_max, self.configs.y_max, self.configs.z_max, self.configs.r_max]
        
        for i, axis in enumerate(checkpoints.T):
            if i ==4: 
                axis[axis<0] += 360
            axis[axis==-1] = MAX_VALUES[i]
            axis[axis>MAX_VALUES[i]] = MAX_VALUES[i]
            
        
        total_dist, sections = self._calcTotalDist(checkpoints)
        # perc = np.array(sections) / total_dist
       
        step_size= round(total_dist/n_imgs)
        
        return step_size
    
    
    def saveData(self, path, overwrite=True):
        
        df = pd.DataFrame(self.data)
        file = os.path.join(path, "pose_info.csv")
        header = ["it", "x_target", "y_target_a", "z_target_a", "r_target_a",
                  "x_read", "y_read", "z_read", "r_read"]

        df.to_csv(file, header=header, index=False)
        
        
    def saveData_(self, path, it):
        
        data_folder = os.path.join(path, "data")
        
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        
        file_name = str(it).zfill(4) + "_data.txt"
        file_path = os.path.join(data_folder, file_name)
        json_object = json.dumps({k: v.tolist() for k,v in self.data_.items()}, indent=4)
        
        with open(file_path, 'w') as f:
            f.write(json_object)
        
        
    def CameraCalibrationMover(self, moves=20, ref=(285,200,250)):
        
            
        x = random.uniform(0,500)
        ret=False
        
        is_outside = True if abs(x-ref[0]) > 175 else False
        in_front = random.choice([0,1])
        
        y = is_outside * random.uniform(0,400) + (1-is_outside) * (in_front*random.uniform(0,25) + (1-in_front)*random.uniform(400,450)) 
        z = random.uniform(0,50)
        
        r = np.arctan2((ref[1]-y),(ref[0]-x))*180/np.pi
        target_pos = (x,y,z,r)
        self.MoveTo(target_pos)
        
        data_list = [self.move_counter] + [x for x in target_pos] + [x for x in self.current_pos]
        self.data.append(data_list)
        self.move_counter +=1
        
        if self.move_counter==30:
            ret=True
        
        return ret, target_pos
    
    
    def MovePlanner(self, cps, n_imgs=50, r_jitter=False, ret=False):
        
            
        # print(checkpoints)
        checkpoints = cps.copy()
        done = False
        n = 0
        while not done:
            
            jitter = 0
            if r_jitter:
                jitter = np.random.choice([1,-1]) * np.random.uniform(3,4)
                jitter = np.array([0,0,0,jitter])
            
            if n==0:
                step_size = self.evalCPs(checkpoints, n_imgs)
                target_pos = checkpoints[0]
                prev_pos = target_pos
                r = checkpoints[0,3]
                checkpoints = checkpoints[1:]
            
            else:
                next_cp = checkpoints[0]
                dist_to_cp = np.linalg.norm(next_cp[:3] - prev_pos[:3])
                
                if dist_to_cp-step_size < 0.5* step_size:
                # if dist_to_cp < step_size:
                    target_pos = next_cp
                    r = checkpoints[0,3]
                    checkpoints = checkpoints[1:]
                    
                    if not len(checkpoints):
                        done = True
                
                else:
                    direction = (next_cp[:3] - prev_pos[:3]) / dist_to_cp
                    step = direction * step_size
                    step = np.append(step, 0)
                    target_pos = prev_pos + step
                 
                
            target_pos[3] = r
            target_pos += jitter
            print(n, target_pos)
            self.checkpoints.append(target_pos)
            prev_pos = target_pos
            n += 1
    
    def Mover(self):
        
        done = False
        cp = self.checkpoints[self.move_counter]
        self.MoveTo(cp)
        
        
        if self.calibrated:
            T_w2c, T_c2w = self.getWorld2Cam()
            c_w = (T_c2w @ np.array([0,0,0,1]))[:-1]
            c_w = np.append(c_w, self.current_pos[-1])
            
            
            self.data_["target"] = np.array(cp)
            self.data_["read"] = np.array(self.current_pos)
            self.data_["cam"] = c_w
            self.data_["T_c2w"] = T_c2w
            self.data_["T_w2c"] = T_w2c
            
        data_list = [self.move_counter] + [x for x in cp] + [x for x in self.current_pos]
        self.data.append(data_list)
        self.move_counter += 1
                
        if self.move_counter == len(self.checkpoints):
            done = True
            
            
        return done
            
        
    def getCam2Arm(self):
        
        
        if not self.calibrated: 
            print("calibrate first.")
            return None
        
        x,y,z,r = self.getPos()
        
        pin2cam = np.array([self.calib_params.x_pin2cam,
                            self.calib_params.y_pin2cam,
                            self.calib_params.z_pin2cam
                            ])
        
        R = Rot.from_euler("z", r, degrees=True)
        pin2cam = R.apply(pin2cam)
        
        arm2pin = np.array([x,
                            y,
                            -z])
                 
        R_c2p = Rot.from_euler("ZYZ", (r, 90+self.calib_params.r_y_pin2cam, -90), degrees=True).as_matrix()
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
        
    
    def getWorld2Arm(self, p_w, p_a):
        

        R_a2w = np.eye(3)
        t_a2w = p_w - p_a
        t_a2w = np.expand_dims(t_a2w, axis=1)
        
        T_a2w = np.hstack([R_a2w, t_a2w])
        T_a2w = np.vstack([T_a2w, np.array([0,0,0,1])])
        
        T_w2a = np.linalg.inv(T_a2w)
        
        return T_a2w, T_w2a
    
    def getWorld2Cam(self):
        
        
        if not self.calibrated: 
            print("calibrate first.")
            return None
        
        T_c2a, T_a2c = self.getCam2Arm()
        T_w2c = T_a2c @ self.T_w2a
        T_c2w = self.T_a2w @ T_c2a 
             
        return T_w2c, T_c2w
        
        
    def calibrate(self, calib_params, mode="full"):
        
        self.calib_params = calib_params
        
        p_w = np.array([self.calib_params.x_world2ref,
                      self.calib_params.y_world2ref,
                      self.calib_params.z_world2ref
                      ])
        
        self.Sleep()
        
        time.sleep(1)
        print("-" * 45)
        print("Align r_axis.")
        print("Press Enter to continue...")
        print("-" * 45)
        input(". . .")
        r_calib = self.CGA_r.GetCurrentPosition()   
        
        self.rot_offset = r_calib 
        
        if r_calib>180:
            self.rot_offset -= 360
        
         
        if mode=="full":
            
            time.sleep(1)
            print("-" * 45)
            print("Find plate for z calib.")
            print("Press Enter to continue...")
            print("-" * 45)
            input(". . .")
            z_calib = -self.CGA_z.GetCurrentPosition()
            
            time.sleep(1)
            print("-" * 45)
            print("Find hole for x and y calib.")
            print("Press Enter to continue...")
            print("-" * 45)
            input(". . .")
            x_calib = self.CGA_x.GetCurrentPosition()
            y_calib = self.CGA_y.GetCurrentPosition()  
            
            
            print("Saving...")
            p_a = np.array([x_calib, y_calib, z_calib])
                              
            # np.savetxt("./calib_info.txt", offset)
            
            self.T_a2w, self.T_w2a = self.getWorld2Arm(p_w, p_a)
       
            out ={}
            out["T_a2w"] = self.T_a2w
            out["T_w2a"] = self.T_w2a
            
            with open("./WorldArm_calibration.txt", "w") as f:
                f.write(json.dumps({k: v.tolist() for k,v in out.items()}, indent=4))
        
        else:
            
            with open("./WorldArm_calibration.txt", "r") as f:
                inp = json.load(f)
                
            self.T_a2w = np.array(inp["T_a2w"])
            self.T_w2a = np.array(inp["T_w2a"])
        
        self.calibrated = True
        self.MoveTo([0.0,0.0,0.0,0.0])
              
            
        
        return self.calibrated
    
    

            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
