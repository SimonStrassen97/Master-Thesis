
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:53:25 2023

@author: SI042101
"""

# import pycolmap
from pathlib import Path

import os
import open3d as o3d
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

PENet_path = '/home/simonst/github/PENet'
if PENet_path not in sys.path:
    sys.path.append(PENet_path)


from utils.worktable_operations import Worktable
from utils.WT_configs import wt1, wt2, wt3, wt4


from utils.general import PCLConfigs, OffsetParameters
from utils.point_cloud_operations import PointCloud
from utils.point_cloud_operations2 import PointCloud2
from utils.general import loadIntrinsics
from utils.process_colmap_output import process_colmap_pcl


import pandas as pd

import torch
torch.cuda.empty_cache()




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


cp1 = ""
cp2 = "/home/simonst/github/results/no_sparsifier/pe_train/model_best.pth.tar"
cp3 = "/home/simonst/github/results/edge_sparsifier/pe_train/model_best.pth.tar"
cp4 = "/home/simonst/github/results/dots_sparsifier/pe_train/model_best.pth.tar"
cp5 = "/home/simonst/github/results/pe_train/model_best.pth.tar"

names = ["wt1", "wt2", "wt3", "wt4"]
cps = [cp1, cp2, cp3, cp4]
stls = [stl1, stl2, stl3, stl4]
paths = [path1, path2, path3, path4]
wts = [wt1, wt2, wt3, wt4]



##################################################################################
#colmap evaluaton



path1_ = "/home/simonst/github/Datasets/wt/raw/20230514_164433" # wt1
path2_ = "/home/simonst/github/Datasets/wt/raw/20230514_173628" # wt2
path3_ = "/home/simonst/github/Datasets/wt/raw/20230522_163051" # wt3
path4_ = "/home/simonst/github/Datasets/wt/raw/20230522_140447" # wt4.2
paths_ = [path1_, path2_, path3_, path4_]
          

grid_size = 0.007
n_pts = 300000
          
# for i, (name,path,stl,wt_dict) in enumerate(zip(names,paths_,stls,wts)):
    
    
#     mean_e = pd.DataFrame(columns=["n_imgs"] + ["-"])
    
#     mean_e.iloc[:,0] = len(os.listdir(os.path.join(path, "img")))
#     adjusted_e = mean_e.copy()
#     missing_e  = mean_e.copy()
#     added_e  = mean_e.copy()
#     e  = mean_e.copy()
#     mcd  = mean_e.copy()
#     mae1  = mean_e.copy()
#     mse1  = mean_e.copy()
#     rmse1  = mean_e.copy()
#     mae2  = mean_e.copy()
#     mse2  = mean_e.copy()
#     rmse2  = mean_e.copy()
#     cts = mean_e.copy()
    
#     out = os.path.join(out_folder, names[i])
#     colmap_out = Path(os.path.join("/home/simonst/github/pycolmap_out/", name))
#     img_dir = Path(os.path.join(path, "img"))
    
#     if not os.path.exists(colmap_out):
#         colmap_out.mkdir()
    
#     mvs_path = colmap_out / "mvs"
    
#     database_path = colmap_out / "database.db"
    
#     start = time.time()
#     pycolmap.extract_features(database_path, img_dir)
#     pycolmap.match_exhaustive(database_path)
#     maps = pycolmap.incremental_mapping(database_path, img_dir, colmap_out)
#     maps[0].write(colmap_out)
#     # dense reconstruction
#     pycolmap.undistort_images(mvs_path, colmap_out, img_dir)
#     pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
#     pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)
    
#     pcd = process_colmap_pcl(mvs_path, path)
#     t = time.time()-start
                    
#     cts.loc[i, "-"] =  t                  
    
#     wt = Worktable()
#     wt.create_model(wt_dict)
      
#     wt.get_ref_wt(path=stl, grid_size=grid_size, n_pts=n_pts)
#     wt.get_recon_wt(pcd)
    
#     # print("Evaluating...")
#     adjusted_e_, e_, missing_e_, added_e_, mean_e_ = wt.evaluate_grids()    
#     cd_, mcd_, recon2ref, ref2recon =  wt.evaluate_pcl()
    
   

#     mean_e.loc[i, "-"] = mean_e_
#     adjusted_e.loc[i, "-"] = adjusted_e_
#     e.loc[i, "-"] = e_
#     missing_e.loc[i, "-"] = missing_e_
#     added_e.loc[i, "-"] = added_e_
#     mcd.loc[i, "-"] = mcd_            
#     mae1.loc[i, "-"] = recon2ref["mae"]
#     mse1.loc[i, "-"] = recon2ref["mse"]
#     rmse1.loc[i, "-"] = recon2ref["rmse"]
#     mae2.loc[i, "-"] = ref2recon["mae"]
#     mse2.loc[i, "-"] = ref2recon["mse"]
#     rmse2.loc[i, "-"] = ref2recon["rmse"]

#     mean_e.to_csv(os.path.join(out, "colmap_mean_e.csv"), index=False)    
#     adjusted_e.to_csv(os.path.join(out, "colmap_adjusted_e.csv"), index=False)
#     e.to_csv(os.path.join(out, "colmap_e.csv"), index=False)
#     missing_e.to_csv(os.path.join(out, "colmap_missing_e.csv"), index=False)
#     added_e.to_csv(os.path.join(out, "colmap_added_e.csv"), index=False)
#     mcd.to_csv(os.path.join(out, "colmap_mcd.csv"), index=False)
#     mae1.to_csv(os.path.join(out, "colmap_mae1.csv"), index=False)
#     mse1.to_csv(os.path.join(out, "colmap__mse1.csv"), index=False)
#     rmse1.to_csv(os.path.join(out, "colmap_rmse1.csv"), index=False)
#     mae2.to_csv(os.path.join(out, "colmap_mae2.csv"), index=False)
#     mse2.to_csv(os.path.join(out, "colmap_mse2.csv"), index=False)
#     rmse2.to_csv(os.path.join(out, "colmap_rmse2.csv"), index=False)
#     cts.to_csv(os.path.join(out, "colmap_cts.csv"), index=False)
    




# n_imgs = [4]
# voxel_size = 0.0
# filters = [True]
# grid_size = 0.007
# n_pts = 300000

   
# pcl_configs = PCLConfigs(outliers=False, 
#                           pre_voxel_size=voxel_size, 
#                           voxel_size=voxel_size,
#                           hp_radius=75,
#                           angle_thresh=75,
#                           std_ratio_stat=2,
#                           nb_points_stat=50,
#                           nb_points_box=6,
#                           box_radius=2*voxel_size,
#                           registration_method="none",
#                           filters=False,
#                           verbose=True,
#                           n_images=4
#                           )

# pcl = PointCloud(pcl_configs)

# start = time.time()
# pcd = pcl.create_multi_view_pcl(path4, n_images=16, resize=True)
# t = time.time()-start


# wt = Worktable()
# wt.create_model(wt4)
  
# wt.get_ref_wt(path=stl4, grid_size=grid_size, n_pts=n_pts)
# wt.get_recon_wt(pcd)

# # print("Evaluating...")
# adjusted_rel_e, rel_e, missing_e, added_e, mean_e = wt.evaluate_grids()
# cd_, mcd_, recon2ref, ref2recon =  wt.evaluate_pcl()



# pcl2 = PointCloud(pcl_configs)


# start = time.time()
# pcd2 = pcl2.create_multi_view_pcl(path4, n_images=16, run_s2d=cp2)
# t = time.time()-start


# wt2 = Worktable()
# wt2.create_model(wt4)
  
# wt2.get_ref_wt(path=stl4, grid_size=grid_size, n_pts=n_pts)
# wt2.get_recon_wt(pcd2)

# # print("Evaluating...")
# adjusted_rel_e2, rel_e2, missing_e2, added_e2, mean_e2 = wt2.evaluate_grids()
# cd_2, mcd_2, recon2ref2, ref2recon2 =  wt2.evaluate_pcl()




# pcl3 = PointCloud(pcl_configs)


# start = time.time()
# pcd3 = pcl3.create_multi_view_pcl(path4, n_images=16)
# t = time.time()-start


# wt3 = Worktable()
# wt3.create_model(wt4)
  
# wt3.get_ref_wt(path=stl4, grid_size=grid_size, n_pts=n_pts)
# wt3.get_recon_wt(pcd3)

# # print("Evaluating...")
# adjusted_rel_e3, rel_e3, missing_e3, added_e3, mean_e3 = wt3.evaluate_grids()
# cd_3, mcd_3, recon2ref3, ref2recon3 =  wt3.evaluate_pcl()


###################################################################33

# eval_time = time.time() 
# print("Depth completion evaluation 2")
 
# n_imgs = [16]
# voxel_size = [0]
# grid_size = 0.007
# n_pts = 300000



# # paths = [path1]
# # wts =[wt1]
# # names = ["wt1"]
# # n_imgs= [1,2]
# # voxel_size =[0.01,0.02]
# # grid_size = 0.0075
# # n_pts = 250000

# cps = [cp5]

# v_missing = pd.DataFrame(columns=["n_imgs"] + [_.split("/")[5] if _ else "None" for _ in cps ])

# v_missing.iloc[:,0] = n_imgs
# v_added = v_missing.copy()
# mcd = v_missing.copy()
# cts  = v_missing.copy()

# for i, (path, stl, wt_dict) in enumerate(zip(paths, stls, wts)):
    
    
#     out = os.path.join(out_folder, names[i])
#     os.makedirs(out, exist_ok=True)
    
        
#     for cp in cps:
            

        
#             for ii, n in enumerate(n_imgs):
                    
#                 torch.cuda.empty_cache()
            
#                 print(f"n_imgs: {n}, cp: {cp}")
    
                    
#                 pcl_configs = PCLConfigs(outliers=False, 
#                                           pre_voxel_size=0, 
#                                           voxel_size=0,
#                                           hp_radius=75,
#                                           angle_thresh=75,
#                                           std_ratio_stat=2,
#                                           nb_points_stat=50,
#                                           nb_points_box=6,
#                                           box_radius=2*0,
#                                           registration_method="none",
#                                           filters=True,
#                                           )
                
                
#                 pcl = PointCloud(pcl_configs)
                
#                 start = time.time()
#                 pcd = pcl.create_multi_view_pcl(path, n_images=n, run_s2d=cp, resize=True)
#                 t = time.time()-start
                
#                 if cp:
#                     col = cp.split("/")[5] 
#                 else:
#                     col = "None"
                    
#                 cts.loc[ii, col] = t           
                
            
            
#                 wt = Worktable()
#                 wt.create_model(wt_dict)
                  
#                 wt.get_ref_wt(path=stl, grid_size=grid_size, n_pts=n_pts)
#                 wt.get_recon_wt(pcd)
                
#                 # print("Evaluating...")
#                 v_missing_, v_added_ = wt.evaluate_grids()    
#                 cd_, mcd_, recon2ref, ref2recon =  wt.evaluate_pcl()
                
               

#                 v_missing.loc[ii, col] = v_missing_
#                 v_added.loc[ii, col] = v_added_
#                 mcd.loc[ii, col] = mcd_   
        
#     v_missing.to_csv(os.path.join(out, "df_2_v_missing.csv"), index=False)   
#     v_added.to_csv(os.path.join(out, "df_2_v_added.csv"), index=False)   
#     mcd.to_csv(os.path.join(out, "df_2_mcd.csv"), index=False) 
#     cts.to_csv(os.path.join(out, "df_2_cts.csv"), index=False)    




        
        
    
    


# print(time.time()-eval_time)




        
        





# ########################################################################################3

# print("Filter evaluation")

# n_imgs = [4,8,16,24,32,48]
# voxel_size = 0.0
# filters = [True, False]
# grid_size = 0.0075
# n_pts = 300000



# mean_e = pd.DataFrame(columns=["n_imgs"] + [str(f) for f in filters])

# mean_e.iloc[:,0] = n_imgs
# adjusted_e = mean_e.copy()
# missing_e  = mean_e.copy()
# added_e  = mean_e.copy()
# e  = mean_e.copy()
# mcd  = mean_e.copy()
# mae1  = mean_e.copy()
# mse1  = mean_e.copy()
# rmse1  = mean_e.copy()
# mae2  = mean_e.copy()
# mse2  = mean_e.copy()
# rmse2  = mean_e.copy()
# cts  = mean_e.copy()


# eval_time = time.time() 
# for i, (path, stl, wt_dict) in enumerate(zip(paths, stls, wts)):
    
    
#     out = os.path.join(out_folder, names[i])
#     os.makedirs(out, exist_ok=True)
    
    
#     for f in filters:
        
 
#         for ii, n in enumerate(n_imgs):
                
        
#             print(f"n_imgs: {n}, filters: {f}")

                
#             pcl_configs = PCLConfigs(outliers=False, 
#                                       pre_voxel_size=voxel_size, 
#                                       voxel_size=voxel_size,
#                                       hp_radius=75,
#                                       angle_thresh=75,
#                                       std_ratio_stat=2,
#                                       nb_points_stat=50,
#                                       nb_points_box=6,
#                                       box_radius=2*voxel_size,
#                                       registration_method="none",
#                                       filters=f
#                                       )
            
            
#             pcl = PointCloud(pcl_configs)
            
#             start = time.time()
#             pcd = pcl.create_multi_view_pcl(path, n_images=n)
#             t = time.time()-start
#             cts.loc[ii, str(f)] = t           
            
        
        
#             wt = Worktable()
#             wt.create_model(wt_dict)
              
#             wt.get_ref_wt(path=stl, grid_size=grid_size, n_pts=n_pts)
#             wt.get_recon_wt(pcd)
            
#             # print("Evaluating...")
#             adjusted_e_, e_, missing_e_, added_e_, mean_e_ = wt.evaluate_grids()    
#             cd_, mcd_, recon2ref, ref2recon =  wt.evaluate_pcl()
                
#             mean_e.loc[ii, str(f)] = mean_e_
#             adjusted_e.loc[ii, str(f)] = adjusted_e_
#             e.loc[ii, str(f)] = e_
#             missing_e.loc[ii, str(f)] = missing_e_
#             added_e.loc[ii, str(f)] = added_e_
#             mcd.loc[ii, str(f)] = mcd_            
#             mae1.loc[ii, str(f)] = recon2ref["mae"]
#             mse1.loc[ii, str(f)] = recon2ref["mse"]
#             rmse1.loc[ii, str(f)] = recon2ref["rmse"]
#             mae2.loc[ii, str(f)] = ref2recon["mae"]
#             mse2.loc[ii, str(f)] = ref2recon["mse"]
#             rmse2.loc[ii, str(f)] = ref2recon["rmse"]
        
#     mean_e.to_csv(os.path.join(out, "filters_mean_e.csv"), index=False)    
#     adjusted_e.to_csv(os.path.join(out, "filters_adjusted_e.csv"), index=False)
#     e.to_csv(os.path.join(out, "filters_e.csv"), index=False)
#     missing_e.to_csv(os.path.join(out, "filters_missing_e.csv"), index=False)
#     added_e.to_csv(os.path.join(out, "filters_added_e.csv"), index=False)
#     mcd.to_csv(os.path.join(out, "filters_mcd.csv"), index=False)
#     mae1.to_csv(os.path.join(out, "filters_mae1.csv"), index=False)
#     mse1.to_csv(os.path.join(out, "filters_mse1.csv"), index=False)
#     rmse1.to_csv(os.path.join(out, "filters_rmse1.csv"), index=False)
#     mae2.to_csv(os.path.join(out, "filters_mae2.csv"), index=False)
#     mse2.to_csv(os.path.join(out, "filters_mse2.csv"), index=False)
#     rmse2.to_csv(os.path.join(out, "filters_rmse2.csv"), index=False)
#     cts.to_csv(os.path.join(out, "filters_cts.csv"), index=False)
    
# print(time.time()-eval_time)





# ##########################################################################################




# print("Voxel size evaluation")

# n_imgs = [4,8,16,24,32, 48]
# voxel_size = [0, 0.001, 0.0025, 0.005, 0.0075, 0.01]
# grid_size = 0.007
# n_pts = 300000



# mean_e = pd.DataFrame(columns=["n_imgs"] + [str(v) for v in voxel_size])

# mean_e.iloc[:,0] = n_imgs
# adjusted_e = mean_e.copy()
# missing_e  = mean_e.copy()
# added_e  = mean_e.copy()
# e  = mean_e.copy()
# mcd  = mean_e.copy()
# mae1  = mean_e.copy()
# mse1  = mean_e.copy()
# rmse1  = mean_e.copy()
# mae2  = mean_e.copy()
# mse2  = mean_e.copy()
# rmse2  = mean_e.copy()
# cts  = mean_e.copy()



# eval_time = time.time() 
# for i, (path, stl, wt_dict) in enumerate(zip(paths, stls, wts)):
    
    
#     out = os.path.join(out_folder, names[i])
#     os.makedirs(out, exist_ok=True)
    
    
#     for v in voxel_size:
        
 
#         for ii, n in enumerate(n_imgs):
                
        
#             print(f"n_imgs: {n}, voxels: {v}")

                
#             pcl_configs = PCLConfigs(outliers=False, 
#                                       pre_voxel_size=v, 
#                                       voxel_size=v,
#                                       hp_radius=75,
#                                       angle_thresh=75,
#                                       std_ratio_stat=2,
#                                       nb_points_stat=50,
#                                       nb_points_box=6,
#                                       box_radius=2*v,
#                                       registration_method="none",
#                                       filters=True
#                                       )
            
            
#             pcl = PointCloud(pcl_configs)
            
#             start = time.time()
#             pcd = pcl.create_multi_view_pcl(path, n_images=n)
#             t = time.time()-start
#             cts.loc[ii, str(v)] = t           
            
        
        
#             wt = Worktable()
#             wt.create_model(wt_dict)
              
#             wt.get_ref_wt(path=stl, grid_size=grid_size, n_pts=n_pts)
#             wt.get_recon_wt(pcd)
            
#             # print("Evaluating...")
#             adjusted_e_, e_, missing_e_, added_e_, mean_e_ = wt.evaluate_grids()    
#             cd_, mcd_, recon2ref, ref2recon =  wt.evaluate_pcl()
            
             
#             mean_e.loc[ii, str(v)] = mean_e_
#             adjusted_e.loc[ii, str(v)] = adjusted_e_
#             e.loc[ii, str(v)] = e_
#             missing_e.loc[ii, str(v)] = missing_e_
#             added_e.loc[ii, str(v)] = added_e_
#             mcd.loc[ii, str(v)] = mcd_            
#             mae1.loc[ii, str(v)] = recon2ref["mae"]
#             mse1.loc[ii, str(v)] = recon2ref["mse"]
#             rmse1.loc[ii, str(v)] = recon2ref["rmse"]
#             mae2.loc[ii, str(v)] = ref2recon["mae"]
#             mse2.loc[ii, str(v)] = ref2recon["mse"]
#             rmse2.loc[ii, str(v)] = ref2recon["rmse"]
        
      
#     mean_e.to_csv(os.path.join(out, "vs_mean_e.csv"), index=False)
#     adjusted_e.to_csv(os.path.join(out, "vs_adjusted_e.csv"), index=False)
#     e.to_csv(os.path.join(out, "vs_e.csv"), index=False)
#     missing_e.to_csv(os.path.join(out, "vs_missing_e.csv"), index=False)
#     added_e.to_csv(os.path.join(out, "vs_added_e.csv"), index=False)
#     mcd.to_csv(os.path.join(out, "vs_mcd.csv"), index=False)
#     mae1.to_csv(os.path.join(out, "vs_mae1.csv"), index=False)
#     mse1.to_csv(os.path.join(out, "vs_mse1.csv"), index=False)
#     rmse1.to_csv(os.path.join(out, "vs_rmse1.csv"), index=False)
#     mae2.to_csv(os.path.join(out, "vs_mae2.csv"), index=False)
#     mse2.to_csv(os.path.join(out, "vs_mse2.csv"), index=False)
#     rmse2.to_csv(os.path.join(out, "vs_rmse2.csv"), index=False)
#     cts.to_csv(os.path.join(out, "vs_cts.csv"), index=False)
    
# print(time.time()-eval_time)





# #########################################################################################

# eval_time = time.time() 
# print("Depth completion evaluation")
 
# n_imgs = [4,8,16,24,32,48]
# voxel_size = [0]
# grid_size = 0.007
# n_pts = 300000

# cps = [cp1, cp2, cp3, cp4]


# # paths = [path1]
# # wts =[wt1]
# # names = ["wt1"]
# # n_imgs= [1,2]
# # voxel_size =[0.01,0.02]
# # grid_size = 0.0075
# # n_pts = 250000


# mean_e = pd.DataFrame(columns=["n_imgs"] + [_.split("/")[5] if _ else "None" for _ in cps ])

# mean_e.iloc[:,0] = n_imgs
# adjusted_e = mean_e.copy()
# missing_e  = mean_e.copy()
# added_e  = mean_e.copy()
# e  = mean_e.copy()
# mcd  = mean_e.copy()
# mae1  = mean_e.copy()
# mse1  = mean_e.copy()
# rmse1  = mean_e.copy()
# mae2  = mean_e.copy()
# mse2  = mean_e.copy()
# rmse2  = mean_e.copy()
# cts  = mean_e.copy()

# for i, (path, stl, wt_dict) in enumerate(zip(paths, stls, wts)):
    
    
#     out = os.path.join(out_folder, names[i])
#     os.makedirs(out, exist_ok=True)
    
    
#     for v in voxel_size:
        
#         for cp in cps:
            

        
#             for ii, n in enumerate(n_imgs):
                    
#                 torch.cuda.empty_cache()
            
#                 print(f"n_imgs: {n}, cp: {cp}")
    
                    
#                 pcl_configs = PCLConfigs(outliers=False, 
#                                           pre_voxel_size=v, 
#                                           voxel_size=v,
#                                           hp_radius=75,
#                                           angle_thresh=75,
#                                           std_ratio_stat=2,
#                                           nb_points_stat=50,
#                                           nb_points_box=6,
#                                           box_radius=2*v,
#                                           registration_method="none",
#                                           filters=True
#                                           )
                
                
#                 pcl = PointCloud(pcl_configs)
                
#                 start = time.time()
#                 pcd = pcl.create_multi_view_pcl(path, n_images=n, run_s2d=cp)
#                 t = time.time()-start
                
#                 if cp:
#                     col = cp.split("/")[5] 
#                 else:
#                     col = "None" 
                    
#                 cts.loc[ii, col] = t           
                
            
            
#                 wt = Worktable()
#                 wt.create_model(wt_dict)
                  
#                 wt.get_ref_wt(path=stl, grid_size=grid_size, n_pts=n_pts)
#                 wt.get_recon_wt(pcd)
                
#                 # print("Evaluating...")
#                 adjusted_e_, e_, missing_e_, added_e_, mean_e_ = wt.evaluate_grids()    
#                 cd_, mcd_, recon2ref, ref2recon =  wt.evaluate_pcl()
                
              

#                 mean_e.loc[ii, col] = mean_e_
#                 adjusted_e.loc[ii, col] = adjusted_e_
#                 e.loc[ii, col] = e_
#                 missing_e.loc[ii, col] = missing_e_
#                 added_e.loc[ii, col] = added_e_
#                 mcd.loc[ii, col] = mcd_            
#                 mae1.loc[ii, col] = recon2ref["mae"]
#                 mse1.loc[ii, col] = recon2ref["mse"]
#                 rmse1.loc[ii, col] = recon2ref["rmse"]
#                 mae2.loc[ii, col] = ref2recon["mae"]
#                 mse2.loc[ii, col] = ref2recon["mse"]
#                 rmse2.loc[ii, col] = ref2recon["rmse"]
        
#     mean_e.to_csv(os.path.join(out, "ml_mean_e.csv"), index=False)    
#     adjusted_e.to_csv(os.path.join(out, "ml_adjusted_e.csv"), index=False)
#     e.to_csv(os.path.join(out, "ml_e.csv"), index=False)
#     missing_e.to_csv(os.path.join(out, "ml_missing_e.csv"), index=False)
#     added_e.to_csv(os.path.join(out, "ml_added_e.csv"), index=False)
#     mcd.to_csv(os.path.join(out, "ml_mcd.csv"), index=False)
#     mae1.to_csv(os.path.join(out, "ml_mae1.csv"), index=False)
#     mse1.to_csv(os.path.join(out, "ml_mse1.csv"), index=False)
#     rmse1.to_csv(os.path.join(out, "ml_rmse1.csv"), index=False)
#     mae2.to_csv(os.path.join(out, "ml_mae2.csv"), index=False)
#     mse2.to_csv(os.path.join(out, "ml_mse2.csv"), index=False)
#     rmse2.to_csv(os.path.join(out, "ml_rmse2.csv"), index=False)
#     cts.to_csv(os.path.join(out, "ml_cts.csv"), index=False)
    


# print(time.time()-eval_time)

# ###################################################################################333

# print("Calibration evaluation")

# n_imgs = [4,8,16,24,32,48]
# voxel_size = 0.0
# filters = [True, False]
# grid_size = 0.0075
# n_pts = 300000
# calibration = [True, False]


# mean_e = pd.DataFrame(columns=["n_imgs"] + [str(f) for f in calibration])

# mean_e.iloc[:,0] = n_imgs
# adjusted_e = mean_e.copy()
# missing_e  = mean_e.copy()
# added_e  = mean_e.copy()
# e  = mean_e.copy()
# mcd  = mean_e.copy()
# mae1  = mean_e.copy()
# mse1  = mean_e.copy()
# rmse1  = mean_e.copy()
# mae2  = mean_e.copy()
# mse2  = mean_e.copy()
# rmse2  = mean_e.copy()
# cts  = mean_e.copy()


# eval_time = time.time() 
# for i, (path, stl, wt_dict) in enumerate(zip(paths, stls, wts)):
    
    
#     out = os.path.join(out_folder, names[i])
#     os.makedirs(out, exist_ok=True)
    
    
#     for c in calibration:
        
 
#         for ii, n in enumerate(n_imgs):
                
        
#             print(f"n_imgs: {n}, calibration: {c}")

#             if not c:
#                 pcl_configs = PCLConfigs(outliers=False, 
#                                           pre_voxel_size=voxel_size, 
#                                           voxel_size=0.001,
#                                           hp_radius=75,
#                                           angle_thresh=75,
#                                           std_ratio_stat=2,
#                                           nb_points_stat=50,
#                                           nb_points_box=6,
#                                           box_radius=2*voxel_size,
#                                           registration_method="plane",
#                                           filters=True,
#                                           verbose=False,
#                                           n_images=n
#                                           )
                
#                 offset_params = OffsetParameters()
                
#                 K, _, _ = loadIntrinsics()
                
#                 pcl = PointCloud2(pcl_configs, offset_params)
                
#                 start = time.time()
#                 pcl.load_PCL_from_depth(path, K)
#                 pcl.ProcessPCL()
#                 pcd = pcl.unified_pcl
#                 t = time.time()-start
                
            
#             if c:  
#                 pcl_configs = PCLConfigs(outliers=False, 
#                                           pre_voxel_size=voxel_size, 
#                                           voxel_size=0.001,
#                                           hp_radius=75,
#                                           angle_thresh=75,
#                                           std_ratio_stat=2,
#                                           nb_points_stat=50,
#                                           nb_points_box=6,
#                                           box_radius=2*voxel_size,
#                                           registration_method="none",
#                                           filters=True,
#                                           verbose=False
#                                           )
                
#                 pcl = PointCloud(pcl_configs)
                
#                 start = time.time()
#                 pcd = pcl.create_multi_view_pcl(path, n_images=n)
#                 t = time.time()-start
                          
#             cts.loc[ii, str(c)] = t           
            
        
        
#             wt = Worktable()
#             wt.create_model(wt_dict)
              
#             wt.get_ref_wt(path=stl, grid_size=grid_size, n_pts=n_pts)
#             wt.get_recon_wt(pcd)
            
#             # print("Evaluating...")
#             adjusted_e_, e_, missing_e_, added_e_, mean_e_ = wt.evaluate_grids()    
#             cd_, mcd_, recon2ref, ref2recon =  wt.evaluate_pcl()
                
            
#             mean_e.loc[ii, str(c)] = mean_e_
#             adjusted_e.loc[ii, str(c)] = adjusted_e_
#             e.loc[ii, str(c)] = e_
#             missing_e.loc[ii, str(c)] = missing_e_
#             added_e.loc[ii, str(c)] = added_e_
#             mcd.loc[ii, str(c)] = mcd_            
#             mae1.loc[ii, str(c)] = recon2ref["mae"]
#             mse1.loc[ii, str(c)] = recon2ref["mse"]
#             rmse1.loc[ii, str(c)] = recon2ref["rmse"]
#             mae2.loc[ii, str(c)] = ref2recon["mae"]
#             mse2.loc[ii, str(c)] = ref2recon["mse"]
#             rmse2.loc[ii, str(c)] = ref2recon["rmse"]
        
#     mean_e.to_csv(os.path.join(out, "calibration_mean_e.csv"), index=False)
#     adjusted_e.to_csv(os.path.join(out, "calibration_adjusted_e.csv"), index=False)
#     e.to_csv(os.path.join(out, "calibration_e.csv"), index=False)
#     missing_e.to_csv(os.path.join(out, "calibration_missing_e.csv"), index=False)
#     added_e.to_csv(os.path.join(out, "calibration_added_e.csv"), index=False)
#     mcd.to_csv(os.path.join(out, "calibration_mcd.csv"), index=False)
#     mae1.to_csv(os.path.join(out, "calibration_mae1.csv"), index=False)
#     mse1.to_csv(os.path.join(out, "calibration_mse1.csv"), index=False)
#     rmse1.to_csv(os.path.join(out, "calibration_rmse1.csv"), index=False)
#     mae2.to_csv(os.path.join(out, "calibration_mae2.csv"), index=False)
#     mse2.to_csv(os.path.join(out, "calibration_mse2.csv"), index=False)
#     rmse2.to_csv(os.path.join(out, "calibration_rmse2.csv"), index=False)
#     cts.to_csv(os.path.join(out, "calibration_cts.csv"), index=False)
    
# print(time.time()-eval_time)














