
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:57:00 2021

@author: antoine
"""

import os
import cv2
import numpy as np



kitti_dir="/home/antoine/KITTI/training/"

kitti_labels=os.path.join(kitti_dir,"label_2")
kitti_imgs=os.path.join(kitti_dir,"image_2")
kitti_calib="/home/antoine/remote_criann/KITTI/processed_training/"
pred_path="/home/antoine/Realtime3DTracking/output/data/"

list_files=[os.path.join(kitti_labels,file) for file in os.listdir(kitti_labels) if file.endswith('txt')]
pred_list=[os.path.join(pred_path,file) for file in os.listdir(pred_path) if file.endswith('txt')]


new_path="/home/antoine/KITTI/training/lol/data/"

for i, path in enumerate(pred_list):
    new_labels=[]
    gt_path=os.path.join(kitti_labels,path.split('/')[-1])
    with open(gt_path) as f:
        gt_labels=[line.split('\n')[0] for line in f.readlines()]

    with open(path) as f:
        pred_labels=[line.split('\n')[0] for line in f.readlines()]
    
#    with open(new_path+path.split('/')[-1],'w') as f:
#        for label in labels:
#            label += " 0.999\n"
#            f.write(label)
