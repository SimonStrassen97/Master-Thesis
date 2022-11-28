# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:00:20 2022

@author: MA012002
"""
import os, sys
# from PyTeMotion import Axis
import PyTeMotion.Axis as Axis

CGA_x = Axis("CGA", "x",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )
CGA_r = Axis("CGA", "r",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )
CGA_y = Axis("CGA", "y",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )
CGA_z = Axis("CGA", "zshort",os.path.realpath('..\src\PyTeMotion\ConfigurationData\RGA\Fluent_CGA_Config.xml') )

CGA_y.StartTeControl()

CGA_x.Initialize()
CGA_y.Initialize()
CGA_z.Initialize()
CGA_r.Initialize()



for x in [600, 1]: 
    CGA_x.MoveTo(x)
    print("------------------------------------------------")
    print(CGA_z.GetCurrentPosition())
    print("------------------------------------------------")
    pass

for y in [450, 1]:
    CGA_y.MoveTo(y)
    print("------------------------------------------------")
    print(CGA_y.GetCurrentPosition())
    print("------------------------------------------------")
    pass

for z in [225, 5]:
    CGA_z.MoveTo(z)
    print("------------------------------------------------")
    print(CGA_z.GetCurrentPosition())
    print("------------------------------------------------")
    pass

for r in [270, -180]: 
    CGA_r.MoveFor(r)
    print("------------------------------------------------")
    print(CGA_z.GetCurrentPosition())
    print("------------------------------------------------")
    pass

for r in [360]: 
    CGA_r.MoveTo(r)
    print("------------------------------------------------")
    print(CGA_z.GetCurrentPosition())
    print("------------------------------------------------")
    pass

