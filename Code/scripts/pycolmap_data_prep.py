
import os
import numpy as np 


import pycolmap
import open3d as o3d
from pathlib import Path

import struct
import sqlite3
import pandas as pd

from pycolmap_utils.read_write_dense import *
from pycolmap_utils.read_write_model import *
from pycolmap_utils.custom_overwrite import *

import matplotlib.pyplot as plt


# path = "/home/simonst/github/pycolmap_out/full_pipeline"
# cameras_file = os.path.join(path, "0", "cameras.bin")
# cameras = read_cameras_binary(cameras_file)

# path = "/home/simonst/github/pycolmap_out/gui_test"
# # cameras_file = os.path.join(path, "sparse", "cameras.bin")
# # cameras = read_cameras_binary(cameras_file)

# depth_map_geo =  os.path.join(path, "stereo", "depth_maps", "0000_img.png.geometric.bin")
# depth_map_photo =  os.path.join(path,"stereo", "depth_maps", "0000_img.png.photometric.bin")

# dmap_geo = read_array(depth_map_geo)
# dmap_photo = read_array(depth_map_photo)

# dmap_photo = dmap_photo.clip(0,20)
# dmap_geo = dmap_geo.clip(0,20)

# fig,ax = plt.subplots(1,2)
# ax[0].imshow(dmap_geo)
# ax[1].imshow(dmap_photo)
# plt.show()


# path = "/home/simonst/github/Datasets/wt/raw/temp/"
# depth_map_geo =  os.path.join(path, "pycolmap", "mvs", "stereo", "depth_maps", "0000_img.png.geometric.bin")
# depth_map_photo =  os.path.join(path, "pycolmap", "mvs", "stereo", "depth_maps", "0000_img.png.photometric.bin")

# dmap_geo = read_array(depth_map_geo)
# dmap_photo = read_array(depth_map_photo)

# dmap_photo = dmap_photo.clip(0,20)
# dmap_geo = dmap_geo.clip(0,20)

# fig,ax = plt.subplots(1,2)
# ax[0].imshow(dmap_geo)
# ax[1].imshow(dmap_photo)
# plt.show()


# def print_db_info(db_path):
#     cnx = sqlite3.connect(db_path)
#     cursor = cnx.cursor()
#     cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#     print(cursor.fetchall())

#     df = pd.read_sql_query("SELECT * FROM cameras", cnx)
#     param_check = struct.unpack("<dddd", df.loc[0, "params"])
#     print(param_check)
#     cnx.close()

# db_path = "/home/simonst/github/Datasets/wt/raw/temp/pycolmap/database.db"
# print_db_info(db_path)

path = Path("/home/simonst/github/Datasets/wt/raw/20230502_204602/")

path = create_temp_folder(path, 25)

prjct_path = Path(os.path.join(path, "pycolmap"))
sparse_path = Path(os.path.join(prjct_path, "sparse"))
# tri_path = Path(os.path.join(prjct_path, "tri"))
img_dir = Path(os.path.join(path, "img"))

prjct_path.mkdir()
sparse_path.mkdir()
# tri_path.mkdir()

mvs_path = prjct_path / "mvs"
database_path = prjct_path / "database.db"

create_cameras_file(path, sparse_path, cam_model="RADIAL")
create_images_file(path, sparse_path)
create_points_file(path, sparse_path)

pycolmap.extract_features(database_path, img_dir)
pycolmap.match_exhaustive(database_path)

recon = pycolmap.Reconstruction(sparse_path)
pycolmap.triangulate_points(recon, database_path, img_dir, sparse_path)

# # dense reconstruction
# # pycolmap.undistort_images(mvs_path, prjct_path, img_dir)
pycolmap.undistort_images(mvs_path, sparse_path, img_dir)
pycolmap.undistort_images(mvs_path, sparse_path, img_dir)
print("undistorted")

rewrite_cfg(mvs_path)
ops = pycolmap.PatchMatchOptions()
ops.depth_max = 100
ops.depth_min = 0.1

pycolmap.patch_match_stereo(mvs_path, options=ops)  # requires compilation with CUDA

depth_map_geo =  os.path.join(path, "pycolmap", "mvs", "stereo", "depth_maps", "0000_img.png.geometric.bin")
depth_map_photo =  os.path.join(path, "pycolmap", "mvs", "stereo", "depth_maps", "0000_img.png.photometric.bin")

dmap_geo = read_array(depth_map_geo)
dmap_photo = read_array(depth_map_photo)

dmap_photo = dmap_photo.clip(0,6)
dmap_geo = dmap_geo.clip(0,6)

fig,ax = plt.subplots(1,2)
ax[0].imshow(dmap_geo)
ax[1].imshow(dmap_photo)
plt.show()

pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)

cameras_file = os.path.join(path, "pycolmap", "mvs", "sparse", "cameras.bin")
images_file = os.path.join(path, "pycolmap", "mvs", "sparse", "images.bin")
points_file = os.path.join(path, "pycolmap", "mvs", "sparse", "points3D.bin")

cameras = read_cameras_binary(cameras_file)
images = read_images_binary(images_file)
points = read_points3D_binary(points_file)

print(cameras)




