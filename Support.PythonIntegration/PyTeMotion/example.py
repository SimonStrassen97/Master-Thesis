# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:00:20 2022

@author: MA012002
"""
import os, sys
from PyTeMotion import Axis


CGA_y = Axis("CGA", "y",os.path.realpath('..\ConfigurationData\RGA\Fluent_CGA_Config.xml') )
CGA_z = Axis("CGA", "zshort",os.path.realpath('..\ConfigurationData\RGA\Fluent_CGA_Config.xml') )

CGA_y.StartTeControl()
CGA_y.Initialize()
CGA_z.Initialize()

for x in [110, 222.2, 333, 222, 111.0, 123.0]:
    CGA_y.MoveTo(x)
    pass



