
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 09:40:29 2020

@author: antoine
"""

import os

file_path="data/3dcent-NS/train.txt"
dir_criann="/app/media/antoine/NVMe/data-NuScenes/"

with open(file_path,'r') as f:
    files=[file.split('\n')[0] for file in f.readlines()]

new_paths=[]

for file in files:
    new_paths.append(os.path.join(dir_criann,*file.split('/')[5:]))
    
    

with open("new_paths_criann.txt","w") as f:
    for file in new_paths:
        f.write(file+'\n')