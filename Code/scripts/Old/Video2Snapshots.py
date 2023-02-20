# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 19:05:36 2022

@author: Simon Strassen
"""

# Importing all necessary libraries
import cv2
import os
import time
import shutil
import numpy as np

def detectBlurryImgs(path, thresh=50, delete=True):
    imgs = os.listdir(path)
    
    for name in imgs:
        file = os.path.join(path,imgs)
        img = cv2.imread(file, 0)
        
        lapl = cv2.Laplacian(img, cv2.CV_64F)
        score = np.var(lapl)
        print(f"Score is {score}.")
        
        if score < thresh:
            print("Deleting img {name}")
            os.remove(file)
            
def _detectBlurryImgs(img, thresh=30, delete=True):
    
    
    blurry = False
    lapl = cv2.Laplacian(img, cv2.CV_64F)
    score = np.var(lapl)
    print(f"Blurry score is {score}.")
    
    if score < thresh:
        blurry = True
    
    return blurry
        
        
        
    
def createSnapshots(path, every_x_sec):
    # Read the video from specified path
    vid = cv2.VideoCapture(path)
    head, tail = os.path.split(path)
    output_folder = os.path.join(head, tail[:-4])
    
    fps = round(vid.get(cv2.CAP_PROP_FPS))
    print(fps)
    
    try:
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)
    
    except:
        os.makedirs(output_folder)
        
    # frame
    currentframe = 0
    
    while (True):
       
        # reading from frame
        ret, frame = vid.read()
    
        if ret:
            
            if (currentframe / fps) % every_x_sec == 0:
                
                blurry = _detectBlurryImgs(frame)
                
                if blurry:
                    print(f"Skipping blurry img...")
                    continue
               
                # if video is still left continue creating images
                name = os.path.join(output_folder, f"frame_{currentframe}.jpg")
                print('Creating...' + name)
        
                # writing the extracted images
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                cv2.imwrite(name, frame)
        
                # increasing counter so that it will
                # show how many frames are created
                
            currentframe += 1
        else:
            break
    
    # Release all space and windows once done
    vid.release()
    cv2.destroyAllWindows()
    
    return output_folder

def main():
    video = "C:/Users\SI042101\ETH\Master_Thesis\Images\Images MT\Video2\IMG_2418.MOV"
    out = createSnapshots(video, every_x_sec=.5)


if __name__ == '__main__':
    main()