
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 15:08:17 2021

@author: antoine
"""

import os

file_path="/home/antoine/KITTI/splits/val.txt"
dir_criann="/media/antoine/NVMe/KITTI/processed_training/"

with open(file_path,'r') as f:
    files=[file.split('\n')[0] for file in f.readlines()]

new_paths=[]

for file in files:
    new_paths.append(os.path.join(dir_criann,file+".jpg"))
    
    

with open("new_paths_criann.txt","w") as f:
    for file in new_paths:
        f.write(file+'\n')