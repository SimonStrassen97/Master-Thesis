
import os
import numpy as np 
import json
import cv2
from scipy.spatial.transform import Rotation as Rot

import pycolmap
import open3d as o3d
from pathlib import Path

import struct
import collections


# with open("/home/simonst/github/pycolmap_out/full_pipeline/images.bin", "rb") as f:
#      a = f.read()
# value = struct.unpack('Q' * (len(a)//8), a)

# b = np.fromfile("/home/simonst/github/pycolmap_out/full_pipeline/images.bin", np.float64)
# print(value)


prjct_path = Path("/home/simonst/github/pycolmap_out/no_sfm")
sparse_path = Path(os.path.join(prjct_path, "sparse"))

path = Path("/home/simonst/github/Datasets/wt/raw/20230417_174256/")
img_dir = Path(os.path.join(path, "img"))

if not os.path.exists(prjct_path):
    prjct_path.mkdir()

mvs_path = prjct_path / "mvs"
database_path = prjct_path / "database.db"

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


class Image(BaseImage):
    def qvec2rotmat(self):
        pass
        # return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

# create cameras.txt
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras



def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images




def create_cameras_file(inp, outp, cam_model="OPENCV"):

    cameras = []
    cameras_file = os.path.join(outp, "cameras.txt")

    img_list = os.listdir(os.path.join(inp, "img"))
    img_list.sort()
    test_img = cv2.imread(os.path.join(inp, "img", img_list[0]), 0)
    h,w = test_img.shape

    with open(os.path.join(inp, "intrinsics.txt")) as f:
        intr = json.load(f)
    intr = intr["color"]

    dist = intr["dist"]
    k1 = dist[0]
    k2 = dist[1]
    p1 = dist[2]
    p2 = dist[3]
    k3 = dist[4]


    if cam_model == "RADIAL":
        cameras = [f'1 {cam_model} {w} {h} {intr["fx"]} {intr["fy"]} {intr["cx"]} {intr["cy"]} {k1} {k2} {p1} {p2}']
    elif cam_model == "OPENCV":
        cameras = [f'1 {cam_model} {w} {h} {intr["fx"]} {intr["fy"]} {intr["cx"]} {intr["cy"]} {k1} {k2} {p1} {p2}']
    elif cam_model == "PINHOLE":
        cameras = [f'1 {cam_model} {w} {h} {intr["fx"]} {intr["fy"]} {intr["cx"]} {intr["cy"]} {k1}']


    with open(cameras_file, "w") as f:
        for line in cameras:
            f.write(str(line) + "\n" )


def create_images_file(inp, outp):
    # create images.txt


    images = []
    images_file = os.path.join(outp, "images.txt")


    img_list = os.listdir(os.path.join(inp, "img"))
    data_list = os.listdir(os.path.join(inp, "data"))
    img_list.sort()
    data_list.sort()


    for i, (img_file, data_file) in enumerate(zip(img_list, data_list)):

        #  IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

        d = os.path.join(inp, "data", data_file)
        # img = os.path.join(inp, "img", img_file)
        with open(d, "r") as f:
            data = json.load(f)
        
        T_w2c = np.array(data["T_w2c"])
        R = T_w2c[:3,:3]
        t = T_w2c[:3,3]
        
        (qx,qy,qz,qw) = Rot.from_matrix(R).as_quat()
        (tx, ty, tz) = t/1000

        camera_id = 1
        image_id = i+1

        line = f'{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {img_file}'
        images.append(line)

    # print(images)
        
    with open(images_file, "w") as f:
        for line in images:
            f.write(str(line) + "\n \n" )

def create_points_file(inp, outp):

    points = []
    points_file = os.path.join(outp, "points3D.txt")

    with open(points_file, "w") as f:
        for line in points:
            f.write(str(line))


def rewrite_cfg(mvs_path):
    
    with open(os.path.join(mvs_path, "stereo/patch-match.cfg"), "r") as f:
     cfg = f.read()


    cfg_new = ""
    prev = ""
    next = ""
    splitted = cfg.split("__auto__, 20")
    for i, line in enumerate(splitted[:-1]):
        cfg_new += line
        next = str(i+1).zfill(4) + "_img.png"
        source = next
        if prev:
            source = prev + ", " + next
        if (i+1)==len(splitted[:-1]):
            source = prev
        cfg_new += source
        prev = str(i).zfill(4) + "_img.png"
        
    with open(os.path.join(mvs_path, "stereo/patch-match.cfg"), "w") as f:
        f.write(cfg_new)


create_cameras_file(path, sparse_path)
create_images_file(path, sparse_path)
create_points_file(path, sparse_path)

pycolmap.extract_features(database_path, img_dir)
pycolmap.match_exhaustive(database_path)
# # maps = pycolmap.incremental_mapping(database_path, img_dir, out)
# # maps[0].write(out)
# # # dense reconstruction
# # pycolmap.undistort_images(mvs_path, prjct_path, img_dir)
pycolmap.undistort_images(mvs_path, sparse_path, img_dir)
print("undistorted")

rewrite_cfg(mvs_path)
# cameras = read_cameras_binary(os.path.join(mvs_path, "sparse", "cameras.bin"))
# images = read_cameras_binary(os.path.join(mvs_path, "sparse", "images.bin"))
# a=0

ops = pycolmap.PatchMatchOptions()
ops.depth_max = 6
ops.depth_min = 0

pycolmap.patch_match_stereo(mvs_path, options=ops)  # requires compilation with CUDA
pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)


