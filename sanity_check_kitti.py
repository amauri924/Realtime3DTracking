
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:00:39 2021

@author: antoine
"""

import os
import cv2
import numpy as np



kitti_dir="/home/antoine/KITTI/training/"

kitti_labels=os.path.join(kitti_dir,"label_2")
kitti_imgs=os.path.join(kitti_dir,"image_2")
kitti_calib="/home/antoine/remote_criann/KITTI/processed_training/"

list_files=[os.path.join(kitti_labels,file) for file in os.listdir(kitti_labels) if file.endswith('txt')]


for i, path in enumerate(list_files):
    img_path=os.path.join(kitti_imgs, path.split('/')[-1].split('.')[0])+'.png'
    calib_path=kitti_calib+path.split('/')[-1].split('.')[0]+'.npy'
    calib=np.load(calib_path)
    with open(path) as f:
        labels=[line.split('\n')[0] for line in f.readlines()]
    img=cv2.imread(img_path)
    
    for label in labels:
        x = float(label.split(' ')[-4])
        y = float(label.split(' ')[-3])
        z = float(label.split(' ')[-2])
        
        h=float(label.split(' ')[-7])
        
        y-=h/2
        
        homo_3d_center=np.array([[x],[y],[z],[1]])
        
        homo_2d_center=(calib@homo_3d_center)[:3,:]
        homo_2d_center/=homo_2d_center[2,:]
        
        x,y=homo_2d_center[:2,0]
        cv2.circle(img,(round(x),round(y)),3,color=(255,0,0))
    cv2.imwrite("sanity/%i.png"%i,img)