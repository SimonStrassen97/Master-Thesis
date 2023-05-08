# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 17:21:14 2022

@author: SI042101
"""

from distutils.core import setup

setup(
    name="recon_source_code",
    version="1.0",
    author="Simon Strassen",
    author_email="simonst@ethz.ch",
    packages=["PyTeMotion", "utils", "pycolmap_utils"],
    package_dir={"": "src"},
)

