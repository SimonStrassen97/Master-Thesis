
import os
import shutil
import cv2
import json
import cv2
import shutil
from scipy.spatial.transform import Rotation as Rot
import numpy as np


def create_temp_folder(path, n_images):

    root = os.path.dirname(path)
    temp = os.path.join(root, "temp")
    temp_depth = os.path.join(temp, "depth")
    temp_data = os.path.join(temp, "data")
    temp_img = os.path.join(temp, "img")

    if os.path.exists(temp):
        shutil.rmtree(temp)
    os.makedirs(temp)
    os.makedirs(temp_depth)
    os.makedirs(temp_data)
    os.makedirs(temp_img)

    depth_folder = os.path.join(path, "depth")
    dfiles = os.listdir(depth_folder)
    dfiles.sort()
    
    
    img_folder = os.path.join(path, "img")
    ifiles = os.listdir(img_folder)
    ifiles.sort()
    
    
    data_folder = os.path.join(path, "data")
    data_files = os.listdir(data_folder)
    data_files.sort()

    intr_source = os.path.join(path, "intrinsics.txt")
    intr_target = os.path.join(temp, "intrinsics.txt")
    shutil.copyfile(intr_source, intr_target)
    
    idx = np.linspace(0, len(dfiles) - 1, n_images).astype(int)


    for i,x in enumerate(idx):

        n = str(i).zfill(4)

        dname = n+"_depth.png"
        dsource = os.path.join(depth_folder, dfiles[x])
        dtarget = os.path.join(temp_depth, dname)

        iname = n+"_img.png"
        isource = os.path.join(img_folder, ifiles[x])
        itarget = os.path.join(temp_img, iname)

        data_name = n+"_data.txt"
        data_source = os.path.join(data_folder, data_files[x])
        data_target = os.path.join(temp_data, data_name)

        shutil.copyfile(dsource, dtarget)
        shutil.copyfile(isource, itarget)
        shutil.copyfile(data_source, data_target)

    return temp

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
        cameras = [f'1 {cam_model} {w} {h} {intr["fx"]} {intr["cx"]} {intr["cy"]} {k1} {k2}']
    elif cam_model == "OPENCV":
        cameras = [f'1 {cam_model} {w} {h} {intr["fx"]} {intr["fy"]} {intr["cx"]} {intr["cy"]} {k1} {k2} {p1} {p2}']
    elif cam_model == "PINHOLE":
        cameras = [f'1 {cam_model} {w} {h} {intr["fx"]} {intr["fy"]} {intr["cx"]} {intr["cy"]}']


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
        # next = str(i+1).zfill(4) + "_img.png"
        # next2 = str(i+2).zfill(4) + "_img.png"
        # next3 = str(i+3).zfill(4) + "_img.png"
        # source = next
        # if prev:
        #     source = prev + ", " + next + ", " + next2 + ", " + next3
        # if (i+3)==len(splitted[:-1]):
        #     source = prev + ", " + next + ", " + next2
        # if (i+2)==len(splitted[:-1]):
        #     source = prev + ", " + next
        # if (i+1)==len(splitted[:-1]):
        #     source = prev
        # cfg_new += source
        # prev = str(i).zfill(4) + "_img.png"
        cfg_new += "__all__"
        
    with open(os.path.join(mvs_path, "stereo/patch-match.cfg"), "w") as f:
        f.write(cfg_new)
