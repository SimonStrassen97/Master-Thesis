# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:53:25 2023

@author: SI042101
"""


# TODO: Color change for missing grids, PCL with color as distance measure
import os
import open3d as o3d
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2


from utils.worktable_operations import Object, Worktable
from utils.WT_configs import wt1, wt2, wt3, wt4


from utils.general import StreamConfigs, PCLConfigs, StreamConfigs, OffsetParameters
from utils.point_cloud_operations import PointCloud
from utils.general import MSE, RMSE, MAE, ME


import pandas as pd




stl1 = "/home/simonst/github/Datasets/wt/wt1.stl"
stl2 = "/home/simonst/github/Datasets/wt/wt2.stl"
stl3 = "/home/simonst/github/Datasets/wt/wt3.stl"
stl4 = "/home/simonst/github/Datasets/wt/wt4.stl"


out_folder = "/home/simonst/github/Datasets/wt/eval"

path1 = "/home/simonst/github/Datasets/wt/raw/20230514_164433" # wt1
path2 = "/home/simonst/github/Datasets/wt/raw/20230514_173628" # wt2
path3 = "/home/simonst/github/Datasets/wt/raw/20230522_163051" # wt3
path4 = "/home/simonst/github/Datasets/wt/raw/20230522_140447" # wt4.2
# path4 = "C:/Users/SI042101/ETH/Master_Thesis/Data/PyData/20230522_124653" # wt4.1


stls = [stl1, stl2, stl3, stl4]
paths = [path1, path2, path3, path4]
wts = [wt1, wt2, wt3, wt4]
names = ["wt1", "wt2", "wt3", "wt4"]




n_imgs = [4,8,16,24,32, 48]
voxel_size = [0, 0.001, 0.0025, 0.005, 0.0075, 0.01]
grid_size = 0.0075
n_pts = 250000



# paths = [path1]
# wts =[wt1]
# names = ["wt1"]
# n_imgs= [1,2]
# voxel_size =[0.01,0.02]
# grid_size = 0.0075
# n_pts = 250000



mcd = pd.DataFrame(columns=["n_imgs"] + [str(v) for v in voxel_size])
mae1 = pd.DataFrame(columns=["n_imgs"] + [str(v) for v in voxel_size])
mse1 = pd.DataFrame(columns=["n_imgs"] + [str(v) for v in voxel_size])
rmse1 = pd.DataFrame(columns=["n_imgs"] + [str(v) for v in voxel_size])
mae2 = pd.DataFrame(columns=["n_imgs"] + [str(v) for v in voxel_size])
mse2 = pd.DataFrame(columns=["n_imgs"] + [str(v) for v in voxel_size])
rmse2 = pd.DataFrame(columns=["n_imgs"] + [str(v) for v in voxel_size])
cts = pd.DataFrame(columns=["n_imgs"] + [str(v) for v in voxel_size])


mcd.iloc[:,0] = n_imgs
mae1.iloc[:,0] = n_imgs
mse1.iloc[:,0] = n_imgs
rmse1.iloc[:,0] = n_imgs
mae2.iloc[:,0] = n_imgs
mse2.iloc[:,0] = n_imgs
rmse2.iloc[:,0] = n_imgs
cts.iloc[:,0] = n_imgs


eval_time = time.time() 
for i, (path, stl, wt_dict) in enumerate(zip(paths, stls, wts)):
    
    
    out = os.path.join(out_folder, names[i])
    os.makedirs(out, exist_ok=True)
    
    
    for v in voxel_size:
        
 
        for ii, n in enumerate(n_imgs):
                
        
            print(f"n_imgs: {n}, voxels: {v}")

                
            pcl_configs = PCLConfigs(outliers=False, 
                                     pre_voxel_size=v, 
                                     voxel_size=v,
                                     hp_radius=75,
                                     angle_thresh=75,
                                     std_ratio_stat=2,
                                     nb_points_stat=50,
                                     nb_points_box=6,
                                     box_radius=2*v,
                                     registration_method="none",
                                     filters=True
                                     )
            
            
            pcl = PointCloud(pcl_configs)
            
            start = time.time()
            pcd = pcl.create_multi_view_pcl(path, n_images=n)
            t = time.time()-start
            cts.loc[ii, str(v)] = t           
            
        
        
            wt = Worktable()
            wt.create_model(wt_dict)
              
            wt.get_ref_wt(path=stl, grid_size=grid_size, n_pts=n_pts)
            wt.get_recon_wt(pcd)
            
            # print("Evaluating...")
            # ret = wt.evaluate_grids()
            cd_, mcd_, recon2ref, ref2recon =  wt.evaluate_pcl()
            
            mcd.loc[ii, str(v)] = mcd_            
            mae1.loc[ii, str(v)] = recon2ref["mae"]
            mse1.loc[ii, str(v)] = recon2ref["mse"]
            rmse1.loc[ii, str(v)] = recon2ref["rmse"]
            mae2.loc[ii, str(v)] = ref2recon["mae"]
            mse2.loc[ii, str(v)] = ref2recon["mse"]
            rmse2.loc[ii, str(v)] = ref2recon["rmse"]
        
    
    mcd.to_csv(os.path.join(out, "vs_mcd.csv"), index=False)
    mae1.to_csv(os.path.join(out, "vs_mae1.csv"), index=False)
    mse1.to_csv(os.path.join(out, "vs_mse1.csv"), index=False)
    rmse1.to_csv(os.path.join(out, "vs_rmse1.csv"), index=False)
    mae2.to_csv(os.path.join(out, "vs_mae2.csv"), index=False)
    mse2.to_csv(os.path.join(out, "vs_mse2.csv"), index=False)
    rmse2.to_csv(os.path.join(out, "vs_rmse2.csv"), index=False)
    cts.to_csv(os.path.join(out, "vscts.csv"), index=False)
    



print(time.time()-eval_time)




 

# path1 = "C:/Users/SI042101/ETH/Master_Thesis/Data/PyData/20230514_164433" # wt1
# path2 = "C:/Users/SI042101/ETH/Master_Thesis/Data/PyData/20230514_173628" # wt2
# path3 = "C:/Users/SI042101/ETH/Master_Thesis/Data/PyData/20230522_163051" # wt3
# path4 = "C:/Users/SI042101/ETH/Master_Thesis/Data/PyData/20230522_140447" # wt4.2
# # path4 = "C:/Users/SI042101/ETH/Master_Thesis/Data/PyData/20230522_124653" # wt4.1
# paths = [path1, path2, path3, path4]
# wts = [wt1, wt2, wt3, wt4]

# n_imgs = [4,8,16,32]
# voxel_size = [0.001, 0.0025, 0.005, 0.0075, 0.01]
# grid_size = 0.0075
# samples = [250000]

# n_imgs= [1,2]
# voxel_size =[0.01, 0.0075]
# grid_size = 0.0075
# samples = [10000, 25000]


# eval_time = time.time() 
# for path, wt_dict in zip(paths, wts):
    
    
#     computation_times = []
#     mae1 = []
#     mse1 = []
#     rmse1 = []
#     mae2 = []
#     mse2 = []
#     rmse2 = []
#     mcd = []
    
#     fig1, ax1 = plt.subplots(1)
#     ax1.set_title("Chamfer Distance")
#     ax1.set_xlabel("n reference pcl samples")
#     ax1.set_ylabel("mm")
#     ax1.legend()
    
#     fig, ax = plt.subplots(1,3)
#     fig.suptitle("Distance Metrics")
#     ax[0].set_title("MAE")
#     ax[1].set_title("MSE")
#     ax[2].set_title("RMSE")
#     ax[0].set_xlabel("n reference pcl samples")
#     ax[0].set_ylabel("mm")
#     plt.tight_layout()

#     # ax.legend()

        
        
    
#     out = os.path.join(path, "eval", f"{n}_imgs")
#     os.makedirs(out, exist_ok=True)
        
#     pcl_configs = PCLConfigs(outliers=False, 
#                              pre_voxel_size=v, 
#                              voxel_size=v,
#                              hp_radius=75,
#                              angle_thresh=75,
#                              std_ratio_stat=2,
#                              nb_points_stat=50,
#                              nb_points_box=6,
#                              box_radius=2*v,
#                              registration_method="none",
#                              filters=True
#                              )
    
    
#     pcl = PointCloud(pcl_configs)
    
#     start = time.time()
#     pcd = pcl.create_multi_view_pcl(path, n_images=n)
#     computation_times.append(time.time()-start)
    
    
#     mean_dists = []
    
#     for n_pts in samples:
        
#         print(f"n_imgs: {n}, n_pts: {n_pts}")

#         wt = Worktable()
#         wt.create_model(wt_dict)
          
#         wt.get_ref_wt(grid_size=grid_size, n_pts=n_pts)
#         wt.get_recon_wt(pcd)
        
#         # print("Evaluating...")
#         # ret = wt.evaluate_grids()
#         cd_, mcd_, recon2ref, ref2recon =  wt.evaluate_pcl()
        
#         mcd.append(mcd_)
#         mae1.append(recon2ref["mae"])
#         mse1.append(recon2ref["mse"])
#         rmse1.append(recon2ref["rmse"])
#         mae2.append(ref2recon["mae"])
#         mse2.append(ref2recon["mse"])
#         rmse2.append(ref2recon["rmse"])
          
    
#     ax1.plot(samples, mcd, label=f"{n} imgs, {v} mm")


# fig1.savefig(os.path.join(out, "chamfer_dist.png"))
# fig2.savefig(os.path.join(out, "comp_times.png"))

    
    
# wt.visualize(model=True)
# wt.visualize(ref=True, model=True)
# wt.visualize(recon=True)
# wt.visualize(model=True, recon=True)
# wt.visualize(diff=True)


# ref = np.array(ret[0])
# rec = np.array(ret[1])


# mse = MSE(ref, rec)
# rmse = RMSE(ref, rec)
# mae = MAE(ref, rec)
# me = ME(ref, rec)

#################################################################33



# path = "C:/Users/SI042101/ETH/Master_Thesis/Data/PyData/20230514_173628"
# pcl_file = os.path.join(path, "pcl_5_imgs.ply")
# pcl = o3d.io.read_point_cloud(pcl_file)


# wt2 = Worktable()
# wt2.create_model(wt2)
    

# start = time.time()
# wt2.get_ref_wt(grid_size=grid_size, n_pts=n_pts)
# print(f"{time.time()-start}s for ref_wt")

# start = time.time()
# wt2.get_recon_wt(pcl)
# print(f"{time.time()-start}s for recon_wt")



# print("Evaluating...")
# ret = wt2.evaluate()


# # wt.visualize(model=True)
# wt2.visualize(ref=True, model=True)
# wt2.visualize(recon=True, model=True)
# wt2.visualize(diff=True)


# ref = np.array(ret[0])
# rec = np.array(ret[1])


# mse = MSE(ref, rec)
# rmse = RMSE(ref, rec)
# mae = MAE(ref, rec)
# me = ME(ref, rec)











