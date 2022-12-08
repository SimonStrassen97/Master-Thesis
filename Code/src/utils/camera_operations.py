# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:57:57 2022

@author: SI042101
"""

import os
import glob
import cv2
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime



# edit JSONEncoder to convert np arrays to list
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    


class Camera():
    def __init__(self, data_dir=None):
            
        self.camera_params = {}
    
    def calibrate(self, input_dir, checkerboard, compute_error=True, save_data=False):
        """
        

        Parameters
        ----------
        checkerboard : dict with checkerboard info: size: Tuple, scale: float
        save_data : bool

        Returns
        -------
        None.

        """
        
        print("---------------------------------")
        print("----------Calibrating------------")
        print("---------------------------------")
        
        size = checkerboard["size"]
        scale = checkerboard["scale"]
            
        CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        objpoints = []
        imgpoints = []
        
        objp = np.zeros((1, size[0] * size[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)*scale
        
        
        image_list = glob.glob(os.path.join(input_dir,"*"))
        
        for i, image in enumerate(image_list):
            
            print(f"{i+1}/{len(image_list)}")
            
            ref = cv2.imread(os.path.join(input_dir, image))
            gray = cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
        
            ret, corners = cv2.findChessboardCorners(gray, size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            if ret == True:
             
               # refining pixel coordinates for given 2d points.
               corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), CRITERIA)
               
               objpoints.append(objp)
               imgpoints.append(corners2)
                
               # img = cv2.drawChessboardCorners(ref.copy(), size, corners2, ret)
              
        
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
         
            
        for variable in ['ret', 'mtx', 'dist', 'rvecs', 'tvecs']:
            self.camera_params[variable] = eval(variable)
            
        if save_data:
            time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"../../Data/CalibrationData/{time_string}_calibration.json"
            self.save_cam_params(output_dir)
          
        if compute_error:
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                mean_error += error
            print( "total error: {}".format(round(mean_error/len(objpoints),3)) )
            
    
    
    
    def undistort(self, img, mode=0, vis=False):
        
        if not self.camera_params:
            raise Exception("Camera params not defined yet.")
                
        h,  w = img.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_params["mtx"], self.camera_params["dist"], (w,h), 1, (w,h))
        self.camera_params["mtx"] = new_camera_mtx
        
        # undistort
        if mode==0:
            corrected_img = cv2.undistort(img, self.camera_params["mtx"], self.camera_params["dist"], None, new_camera_mtx)
        
        elif mode==1:
            mapx, mapy = cv2.initUndistortRectifyMap(self.camera_params["mtx"], self.camera_params["dist"], new_camera_mtx, (w,h), 5)
            corrected_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            # corrected_img = cv2.remap(img, mapx, mapy, cv2.INTER_LANCZOS4)
        
        else:
            raise ValueError("Mode values 0 or 1.")

        # crop the image
        x, y, w, h = roi
        corrected_img = corrected_img[y:y+h, x:x+w]
        
        if vis:
            fig, ax = plt.subplots(1,2)
            plt.suptitle("Calibration")
            ax[0].imshow(img)
            ax[0].set_title("Distorted")
            ax[1].imshow(corrected_img)
            ax[1].set_title("Corrected")
            
        return corrected_img                               
    
    def save_cam_params(self, path):
        with open(path, 'w') as f:
            if self.camera_params:
                json.dump(self.camera_params, f, indent=4, cls=NumpyEncoder)
            else:
                print("No params to be saved.")
            f.close()
        
    
    
    def load_cam_params(self, path):
        with open(path, 'r') as f:
            self.camera_params = json.load(f)
            f.close()
        
        self.camera_params["mtx"] = np.array(self.camera_params["mtx"])
        self.camera_params["dist"] = np.array(self.camera_params["dist"])
        self.camera_params["rvecs"] = (np.array(self.camera_params["rvecs"][0]),self.camera_params["rvecs"][1])
        self.camera_params["tvecs"] = (np.array(self.camera_params["tvecs"][0]),self.camera_params["tvecs"][1])
  
        return self.camera_params
      

class PseudoStereoCamera(Camera):
    def __init__(self):
            super().__init__()
    
    
    def calibrate(self, input_dir, checkerboard):
        
        size = checkerboard["size"]
        scale = checkerboard["scale"]
            
        CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        
        objpoints = []
        imgpointsL = []
        imgpointsR = []
        imgpoints = []
        
        objp = np.zeros((1, size[0] * size[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)*scale
        
        
        image_list = glob.glob(os.path.join(input_dir,"*"))
        
        for i, image in enumerate(image_list):
            
            print(f"{i+1}/{len(image_list)}")
            
            ref = cv2.imread(os.path.join(input_dir, image))
            gray = cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
            self.img_size=gray.shape
        
            ret, corners = cv2.findChessboardCorners(gray, size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            if ret == True:
             
               # refining pixel coordinates for given 2d points.
               corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), CRITERIA)
               
               
               objpoints.append(objp)
               imgpoints.append(corners2)
               if "left" in image:
                   imgpointsL.append(corners2)
               elif "right" in image:
                   imgpointsR.append(corners2)
            
                
            # self.control = cv2.drawChessboardCorners(ref.copy(), size, corners2, ret)
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        w, h = gray.shape
        mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h),1,(w,h))
        
        for variable in ['ret', 'mtx', 'dist', 'rvecs', 'tvecs']:
            self.camera_params[variable] = eval(variable)
            
    
    def undistort(self, imgR, imgL):
        
        if not self.camera_params:
            raise Exception("Camera params not defined yet.")
                
        h,  w = imgR.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_params["mtx"], self.camera_params["dist"], (w,h), 1, (w,h))
        self.camera_params["mtx"] = new_camera_mtx
        
        (leftRectification, rightRectification, leftProjection, rightProjection,
         dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(self.camera_params["mtx"],
                                                                     self.camera_params["dist"], 
                                                                     self.camera_params["mtx"], 
                                                                     self.camera_params["dist"],
                                                                     self.img_size, 
                                                                     self.rotationMatrix,
                                                                     self.translationVector)
        
        stereoMapL_x, stereoMapL_y = cv2.initUndistortRectifyMap(self.camera_params["mtx"],
                                                                 self.camera_params["dist"],
                                                                 leftRectification, leftProjection,
                                                                 self.img_size, cv2.CV_32FC1)
        
        stereoMapR_x, stereoMapR_y = cv2.initUndistortRectifyMap(self.camera_params["mtx"], 
                                                                 self.camera_params["dist"], 
                                                                 rightRectification,
                                                                 rightProjection, 
                                                                 self.img_size, cv2.CV_32FC1)

        imgR = cv2.remap(imgR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        imgL = cv2.remap(imgL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
        return imgR, imgL
    
    
        
    def stereoUndistort(self, img1, img2, R=None, T=None):
        
        if not self.camera_params:
            raise Exception("Camera params not defined yet.")
            
        h,  w = self.img_size
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_params["mtx"], self.camera_params["dist"], (w,h), 1, (w,h))
        self.camera_params["mtx"] = new_camera_mtx
        
        if not R or not T:
            pts1, pts2 = self._findMatches(img1, img2)
            
            E, inliers = cv2.findEssentialMat(pts1 , pts2, self.camera_params["mtx"], cv2.FM_RANSAC, prob=0.999, threshold=2)
            
            pts1 = pts1[inliers.ravel() == 1]
            pts2 = pts2[inliers.ravel() == 1]
            
            ret, R, T, _ = cv2.recoverPose(E, pts1, pts2, self.camera_params["mtx"])
            
        
        
        (Rect1, Rect2, P1, P2, Q, ROI1, ROI2) = cv2.stereoRectify(self.camera_params["mtx"],
                                                                  self.camera_params["dist"], 
                                                                  self.camera_params["mtx"], 
                                                                  self.camera_params["dist"],
                                                                  self.img_size, 
                                                                  R,
                                                                  T)
            
        stereoMap1_x, stereoMap1_y = cv2.initUndistortRectifyMap(self.camera_params["mtx"],
                                                                 self.camera_params["dist"],
                                                                 Rect1, P1,
                                                                 self.img_size, cv2.CV_32FC1)
        
        stereoMap2_x, stereoMap2_y = cv2.initUndistortRectifyMap(self.camera_params["mtx"], 
                                                                 self.camera_params["dist"], 
                                                                 Rect2, P2, 
                                                                 self.img_size, cv2.CV_32FC1)

        img1 = cv2.remap(img1, stereoMap1_x, stereoMap1_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        img2 = cv2.remap(img2, stereoMap2_x, stereoMap2_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        
        return img1, img2
    
        
        
        
    
    def _findMatches(self, img1, img2, feature="sift", matcher="flann", vis=False):
        
                
        # (sift, orb, fast-brief, star-brief)
        kp1, des1 = self._extractFeatures(img1, feature=feature)
        kp2, des2 = self._extractFeatures(img2, feature=feature)        
        if vis:
            kp_img1 = cv2.drawKeypoints(img1.copy(), kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            kp_img2 = cv2.drawKeypoints(img2.copy(), kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
     
            kp_fig, kp_ax  = plt.subplots(1,2)
            kp_ax[0].imshow(kp_img1)
            kp_ax[1].imshow(kp_img2)
        
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        matchesMask = [[0, 0] for i in range(len(matches))]
        good = []
        pts1 = []
        pts2 = []
        
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                # Keep this keypoint pair
                matchesMask[i] = [1, 0]
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
        
        # Draw the keypoint matches between both pictures
        # Still based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
        if vis:
            n_ = min(len(matches), 400)
            idx = [random.randint(0,len(matches)) for i in range(n_)]
            matchesMask_= []
            matches_ = []
            for i in idx:
                matchesMask_.append(matchesMask[i])
                matches_.extend(list((matches[i],)))
                
            matches_ = tuple(matches_)
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask_,
                               flags=cv2.DrawMatchesFlags_DEFAULT)
            
            keypoint_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches_, None, **draw_params)
            matches_fig = plt.figure()
            plt.imshow(keypoint_matches)
     
        return np.array(pts1), np.array(pts2)    
    
    
    
    def _extractFeatures(self, img, feature="sift"):
        
        if feature == "sift":
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(img, None)
            
        elif feature == "orb":
            orb = cv2.ORB_create()
            kp, des = orb.detectAndCompute(img, None)
            
        elif feature == "surf":
            raise ValueError("Surf is patented")
            surf = cv2.xfeatures2d.SURF_create(400)
            kp, des = surf.detectAndCompute(img, None)
            
        elif feature == "fast-brief":
            fast = cv2.FastFeatureDetector_create()
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            
            kp = fast.detect(img, None)
            kp, des = brief.compute(img, kp)
            
        
        elif feature == "star-brief":
            star = cv2.xfeatures2d.StarDetector_create()
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            
            kp = star.detect(img, None)
            kp, des = brief.compute(img, kp)
            
        else:
            raise ValueError("InvalidInput")
            
        
        return kp, des
            
         
    
    
class StereoCamera(Camera):
    def __init__(self):
            super().__init__()
    
    
    def calibrate(self, input_dir, checkerboard):
        
        size = checkerboard["size"]
        scale = checkerboard["scale"]
            
        CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        objpoints = []
        imgpointsL = []
        imgpointsR = []
        imgpoints = []
        
        objp = np.zeros((1, size[0] * size[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)*scale
        
        
        image_list = glob.glob(os.path.join(input_dir,"*"))
        
        for i, image in enumerate(image_list):
            
            print(f"{i+1}/{len(image_list)}")
            
            ref = cv2.imread(os.path.join(input_dir, image))
            gray = cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
            self.img_size=gray.shape
        
            ret, corners = cv2.findChessboardCorners(gray, size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            if ret == True:
             
               # refining pixel coordinates for given 2d points.
               corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), CRITERIA)
               
               
               objpoints.append(objp)
               imgpoints.append(corners2)
               if "left" in image:
                   imgpointsL.append(corners2)
               elif "right" in image:
                   imgpointsR.append(corners2)
            
                
            self.control = cv2.drawChessboardCorners(ref.copy(), size, corners2, ret)
              
        
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        w, h = gray.shape
        mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h),1,(w,h))
        
        for variable in ['ret', 'mtx', 'dist', 'rvecs', 'tvecs']:
            self.camera_params[variable] = eval(variable)
            
        (_, _, _, _, _, self.rotationMatrix, self.translationVector, _, _) = cv2.stereoCalibrate(objp, imgpointsL, imgpointsR,
        mtx, dist,
        mtx, dist,
        gray.shape, None, None, None, None,
        cv2.CALIB_FIX_INTRINSIC, CRITERIA)
        
        self.ret = cv2.stereoCalibrate(objp, imgpointsL, imgpointsR,
        mtx, dist,
        mtx, dist,
        gray.shape, None, None, None, None,
        cv2.CALIB_FIX_INTRINSIC, CRITERIA)
    
        
        
    
    def undistort(self, imgR, imgL):
        
        if not self.camera_params:
            raise Exception("Camera params not defined yet.")
                
        h,  w = imgR.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_params["mtx"], self.camera_params["dist"], (w,h), 1, (w,h))
        self.camera_params["mtx"] = new_camera_mtx
        
        (leftRectification, rightRectification, leftProjection, rightProjection,
         dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(self.camera_params["mtx"],
                                                                     self.camera_params["dist"], 
                                                                     self.camera_params["mtx"], 
                                                                     self.camera_params["dist"],
                                                                     self.img_size, 
                                                                     self.rotationMatrix,
                                                                     self.translationVector)
        
        stereoMapL_x, stereoMapL_y = cv2.initUndistortRectifyMap(self.camera_params["mtx"],
                                                                 self.camera_params["dist"],
                                                                 leftRectification, leftProjection,
                                                                 self.img_size, cv2.CV_32FC1)
        
        stereoMapR_x, stereoMapR_y = cv2.initUndistortRectifyMap(self.camera_params["mtx"], 
                                                                 self.camera_params["dist"], 
                                                                 rightRectification,
                                                                 rightProjection, 
                                                                 self.img_size, cv2.CV_32FC1)

        imgR = cv2.remap(imgR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        imgL = cv2.remap(imgL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
        return imgR, imgL
   

     
# class CameraMono(Camera):
#     def __init__(self):
#         super().__init__()
    
#     def getAbsoluteDepth(self, img, depth_img, checkerboard):
        
#         self._get_pxpermm(img, checkerboard)
#         stamp = self._createStamp()
#         idx = self._findCross(img, stamp)
        
#         pass
        
    
#     def _get_pxpermm(self, img, checkerboard):
        
#         size = checkerboard["size"]
#         scale = checkerboard["scale"]
        
#         CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
#         ret, corners = cv2.findChessboardCorners(img, size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
#         if ret == True:
          
#             # refining pixel coordinates for given 2d points.
#             corners = cv2.cornerSubPix(img, corners, (11,11),(-1,-1), CRITERIA)
            
#             corners = corners.reshape(-1, size[0], 2)
                       
                       
#             diff_x = np.linalg.norm(corners-np.roll(corners,-1,axis=1),axis=2)[:,:-1]
#             diff_y = np.linalg.norm(corners-np.roll(corners,-1,axis=0),axis=2)[:-1]
            
#             diff = np.concatenate([diff_x.reshape(-1),diff_y.reshape(-1)])
            
#             avg_diff = np.mean(diff)
#             self.pxpermm = avg_diff/scale            
                      
    
#     def _createStamp(self):
                        
#         circle = 2.5*self.pxpermm
#         cross = 4.5*self.pxpermm 
#         thickness = 0.2*self.pxpermm
                
#         dim = int(cross+5)
        
#         stamp = np.full((dim,dim), 255, dtype = np.uint8)
#         stamp[int(dim/2)-int(cross/2):int(dim/2)+int(cross/2), int(dim/2)-int(thickness/2):int(dim/2)+int(thickness/2)] = 0
#         stamp[int(dim/2)-int(thickness/2):int(dim/2)+int(thickness/2), int(dim/2)-int(cross/2):int(dim/2)+int(cross/2)] = 0
#         cv2.circle(stamp, (int(dim/2), int(dim/2)), int(circle/2), 0, int(thickness))
        
#         return stamp
        
#     def _findCross(self, img, stamp):
        
#         _,thresh = cv2.threshold(img, 130, 255 ,cv2.THRESH_BINARY)
#         blurred_img = cv2.GaussianBlur(thresh, (3,3), 0)
#         blurred_stamp = cv2.GaussianBlur(stamp, (3,3), 0)

#         cc = cv2.matchTemplate(blurred_img, blurred_stamp,  method=cv2.TM_CCOEFF_NORMED)
        
#         max_val = cc.max()
#         idx = np.where(cc==max_val)
        
#         return idx
    
    
    
    
    