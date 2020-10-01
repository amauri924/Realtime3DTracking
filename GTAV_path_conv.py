
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:14:35 2020

@author: antoine
"""

import os

file_path="/home/antoine/remote_criann/GTA_Dataset/train_0.txt"
dir_criann="/save/2020010/amauri03/GTA_Preprocessed/"

with open(file_path,'r') as f:
    files=[file.split(' ')[0] for file in f.readlines()]

new_paths=[]

for file in files:
    new_paths.append(os.path.join(dir_criann,*file.split('/')[-3:]))
    
    

with open("new_paths_criann.txt","w") as f:
    for file in new_paths:
        f.write(file+'\n')