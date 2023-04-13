


import pycolmap
import open3d as o3d
import os
import numpy as np
from pathlib import Path

print(pycolmap.has_cuda)


out = Path("/home/simonst/github/pycolmap_out/")
img_dir = Path("/home/simonst/github/Datasets/wt/raw/20230227_181401/img/")

if not os.path.exists(out):
    out.mkdir()
mvs_path = out / "mvs"
print(mvs_path)
database_path = out / "database.db"

# pycolmap.extract_features(database_path, img_dir)
# pycolmap.match_exhaustive(database_path)
# maps = pycolmap.incremental_mapping(database_path, img_dir, out)
# maps[0].write(out)
# # dense reconstruction
# pycolmap.undistort_images(mvs_path, out, img_dir)
# pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
# pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)

pcd = o3d.io.read_point_cloud(os.path.join(mvs_path, "dense.ply"))
pts = np.array(pcd.points)
cond = np.logical_and(abs(pts[:,2])<100 ,abs(pts[:,0])<100, abs(pts[:,1])<100)
ind = np.where(cond)[0]
pcd = pcd.select_by_index(ind, invert=False)

origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=np.array([0., 0., 0.]))
o3d.visualization.draw_geometries([pcd, origin])

a=0