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
from utils.WT_configs import *


from utils.general import StreamConfigs, PCLConfigs, StreamConfigs, OffsetParameters
from utils.point_cloud_operations import PointCloud
from utils.general import MSE, RMSE, MAE, ME



# wt = Worktable()
# wt.create_model(wt3)
# wt.visualize(model=True)



out_folder = "C:/Users/SI042101/ETH/Master_Thesis/Data/PyData/eval"



path1 = "C:/Users/SI042101/ETH/Master_Thesis/Data/PyData/20230514_164433" # wt1
path2 = "C:/Users/SI042101/ETH/Master_Thesis/Data/PyData/20230514_173628" # wt2
path3 = "C:/Users/SI042101/ETH/Master_Thesis/Data/PyData/20230522_163051" # wt3
path4 = "C:/Users/SI042101/ETH/Master_Thesis/Data/PyData/20230522_140447" # wt4.2
# path4 = "C:/Users/SI042101/ETH/Master_Thesis/Data/PyData/20230522_124653" # wt4.1


paths = [path1, path2, path3, path4]
wts = [wt1, wt2, wt3, wt4]
names = ["wt1", "wt2", "wt3", "wt4"]
n_imgs = [4,8,16,24,32, 48]
voxel_size = [0, 0.001, 0.0025, 0.005, 0.0075, 0.01]
grid_size = 0.0075



# paths = [path1]
# wts =[wt1]
# names = ["wt1"]
# n_imgs= [1,2]
# voxel_size =[0.01,0.02]
# grid_size = 0.0075
n_pts = 250000



eval_time = time.time() 
for i, (path, wt_dict) in enumerate(zip(paths, wts)):
    
    
    out = os.path.join(out_folder, names[i])
    os.makedirs(out, exist_ok=True)
    
    
    fig1, ax1 = plt.subplots(1)
    ax1.set_title("Chamfer Distance D(v=V,n)")
    ax1.set_xlabel("n images")
    ax1.set_ylabel("mm")
    plt.tight_layout()
    
    
    
    fig2, ax2 = plt.subplots(1)
    ax2.set_title("Computation Times T(v=V,n)")
    ax2.set_xlabel("n images")
    ax2.set_ylabel("s")
    plt.tight_layout()
    
        # ax.legend()
  
    fig3, ax3 = plt.subplots(1,3)
    fig3.suptitle("Recon2Ref: Distance Metrics")
    ax3[0].set_title("MSE")
    ax3[1].set_title("RMSE")
    ax3[2].set_title("MAE")
    ax3[0].set_xlabel("n images")
    ax3[0].set_ylabel("mm")
    plt.tight_layout()
    
    fig4, ax4 = plt.subplots(1,3)
    fig4.suptitle("Ref2Recon: Distance Metrics")
    ax4[0].set_title("MSE")
    ax4[1].set_title("RMSE")
    ax4[2].set_title("MAE")
    ax4[0].set_xlabel("n images")
    ax4[0].set_ylabel("mm")
    plt.tight_layout()
    
    for v in voxel_size:
        
        cts = []
        mae1 = []
        mse1 = []
        rmse1 = []
        mae2 = []
        mse2 = []
        rmse2 = []
        mcd = []
        
       
               
        for n in n_imgs:
                
        
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
            cts.append(time.time()-start)
        
        
            wt = Worktable()
            wt.create_model(wt_dict)
              
            wt.get_ref_wt(grid_size=grid_size, n_pts=n_pts)
            wt.get_recon_wt(pcd)
            
            # print("Evaluating...")
            # ret = wt.evaluate_grids()
            cd_, mcd_, recon2ref, ref2recon =  wt.evaluate_pcl()
            
            mcd.append(mcd_)
            mae1.append(recon2ref["mae"])
            mse1.append(recon2ref["mse"])
            rmse1.append(recon2ref["rmse"])
            mae2.append(ref2recon["mae"])
            mse2.append(ref2recon["mse"])
            rmse2.append(ref2recon["rmse"])
       
        ax1.plot(n_imgs, mcd, label=f"{v} mm")
        ax2.plot(n_imgs, cts, label=f"{v} mm")
        ax3[0].plot(n_imgs, mse1, label=f"{v} mm")
        ax3[1].plot(n_imgs, rmse1, label=f"{v} mm")
        ax3[2].plot(n_imgs, mae1, label=f"{v} mm")
        ax4[0].plot(n_imgs, mse2, label=f"{v} mm")
        ax4[1].plot(n_imgs, rmse2, label=f"{v} mm")
        ax4[2].plot(n_imgs, mae2, label=f"{v} mm")
    
    
    
    ymin = min(ax3[1].get_ylim() + ax3[2].get_ylim())
    ymax = min(ax3[1].get_ylim() + ax3[2].get_ylim())
    ax3[1].set_ylim(ymin,ymax)
    ax3[2].set_ylim(ymin,ymax)
    
    ymin = min(ax4[1].get_ylim() + ax4[2].get_ylim())
    ymax = min(ax4[1].get_ylim() + ax4[2].get_ylim())
    ax4[1].set_ylim(ymin,ymax)
    ax4[2].set_ylim(ymin,ymax)

    

    ax1.legend()
    ax2.legend()
    
    handles, labels = ax3[0].get_legend_handles_labels()
    fig3.legend(handles, labels, loc=7)
    fig3.tight_layout()
    fig3.subplots_adjust(right=0.75)
    
    handles, labels = ax4[0].get_legend_handles_labels()
    fig4.legend(handles, labels, loc=7)
    fig4.tight_layout()
    fig4.subplots_adjust(right=0.75)
    
    fig1.savefig(os.path.join(out, "chamfer_dist.png"))
    fig2.savefig(os.path.join(out, "comp_times.png"))
    fig3.savefig(os.path.join(out, "Recon2Ref.png"))
    fig4.savefig(os.path.join(out, "Ref2Recon.png"))
    
    
    


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











