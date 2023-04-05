# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:53:17 2022

@author: SI042101
"""


import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
# import pandas as pd

from utils.axis_operations import AxisMover
from utils.general import AxisConfigs, CalibParams

cv2.destroyAllWindows()


calibrate=True
vis=True
output_dir = "C:\\Users\SI042101\ETH\Master_Thesis\Data\PyData"

time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
data_output_folder = os.path.join(output_dir, time_string)
os.makedirs(data_output_folder)




from PyTeMotion.AxisFunctions import Axis
CGA_x = Axis("CGA", "x",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )
CGA_r = Axis("CGA", "r",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )
CGA_y = Axis("CGA", "y",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )
CGA_z = Axis("CGA", "zshort",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )


CGA_y.StartTeControl()


CGA_z.Initialize()
CGA_r.Initialize()
CGA_x.Initialize()
CGA_y.Initialize()
    


#####################################################3

# Mover init

configs = AxisConfigs()
axis = AxisMover(CGA_x, CGA_y, CGA_z, CGA_r, configs)


calib_params = CalibParams()
T_arm, T_cam = AxisMover.calibrate(calib_params)
















