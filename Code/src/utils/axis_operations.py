# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:01:02 2022

@author: SI042101
"""

import os
import numpy as np
import pandas as pd
import cv2

class AxisMover():
    def __init__(self, x, y, z, r, configs):
        # Axes
        self.CGA_x = x
        self.CGA_y = y
        self.CGA_z = z
        self.CGA_r = r
        
        self.checkpoints = None
        self.step_size = None
        
        self.configs = configs
        
        # ouput data
        self.data = []
        
        # Axis state
        self.move_counter = 0
        
        self._updatePos()

    
    
    def _updatePos(self, update=False):
        
        
        self.current_pos =  np.round(np.array([self.CGA_x.GetCurrentPosition(),
                             self.CGA_y.GetCurrentPosition(),
                             self.CGA_z.GetCurrentPosition(),
                             self.CGA_r.GetCurrentPosition()]),3)
        
        
        
    def MoveTo(self, dest):
        
        (x,y,z,r) = dest
        self.CGA_x.MoveTo(x)
        self.CGA_y.MoveTo(y)
        self.CGA_z.MoveTo(z)
        
        
        r_0 = self.current_pos[3]
        
        if round(r_0,1) > 180:
            r_0 -= 360
        if round(r,1) > 180:
            r -= 360
        
        self.rr = (r_0, r)
        
        if round(r_0,1) != round(r,1):

            self.CGA_r.MoveFor(-r_0)
            self.CGA_r.MoveFor(r)
            
            
    def MoveFor(self, dist):
        
        (x,y,z,r) = dist
        self.CGA_x.MoveFor(x)
        self.CGA_y.MoveFor(y)
        self.CGA_z.MoveFor(z)
        self.CGA_r.MoveFor(r)
   
    
    def _calcTotalDist(self, cps):
        
        total_dist = 0
        prev_cp = None
        for i, cp in enumerate(cps[:,0:3]):
            if i!=0:
                total_dist += np.linalg.norm(cp-prev_cp)
            prev_cp = cp
            
        return total_dist
            
        
        
    def evalCPs(self, checkpoints):
        
        self.checkpoints=checkpoints
        
        n_imgs = self.configs.n_images
        
        MAX_VALUES = [self.configs.x_max, self.configs.y_max, self.configs.z_max, self.configs.r_max]
        
        for i, axis in enumerate(self.checkpoints.T):
            if i ==4: 
                axis[axis<0] += 360
            axis[axis==-1] = MAX_VALUES[i]
            axis[axis>MAX_VALUES[i]] = MAX_VALUES[i]
            
        
        total_dist = self._calcTotalDist(checkpoints)
        
        self.step_size= round(total_dist/n_imgs)
        
    def saveData(self, path, overwrite=True):
        
        df = pd.DataFrame(self.data)
        file = os.path.join(path, "pose_info.csv")
        header = ["it", "x_target", "y_target", "z_target", "r_target", "x_read", "y_read", "z_read", "r_read"]

        df.to_csv(file, header=header, index=False)
        
    def MovePlanner(self, checkpoints, ret=False):
        
        done = False
        
        if self.move_counter==0:
            self.MoveTo(checkpoints[0])
            self.evalCPs(checkpoints)
            self._updatePos()
          
        
        next_cp = self.checkpoints[0]
        dist_to_cp = np.linalg.norm(next_cp[:3] - self.current_pos[:3])
        
        if dist_to_cp-self.step_size < 0.5*self.step_size:
            target_pos = next_cp
            self.checkpoints = self.checkpoints[1:]
            
            if not len(self.checkpoints):
                done = True
        
        else:
            direction = (next_cp[:3] - self.current_pos[:3]) / dist_to_cp
            step = direction * self.step_size
            step = np.append(step, 0)
            target_pos = self.current_pos + step
        
        # move
        self.MoveTo(target_pos)
        
        # update state
        self.move_counter += 1
        self._updatePos()
        error = target_pos[:3]-self.current_pos[:3]
        
        
        # add to data
        data_list = [self.move_counter] + [x for x in target_pos] + [x for x in self.current_pos]
        self.data.append(data_list)
        
        if ret:
            return done, target_pos, error
        else:
            return done
        
        
        
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
