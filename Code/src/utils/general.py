# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 15:08:35 2021

@author: SI042101
"""


import cv2
import numpy as np



########################################################################################################
# Functions for Image formating #
########################################################################################################

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

    
    
    