
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:14:35 2020

@author: antoine
"""

import os

file_path="/home/antoine/Realtime3DTracking/data/GTA_3dcent/test_0.txt"
dir_criann="/home/antoine/remote_criann/GTA_Preprocessed_v2/"

with open(file_path,'r') as f:
    files=[file.split(' ')[0] for file in f.readlines()]

new_paths=[]

for file in files:
    new_paths.append(os.path.join(dir_criann,*file.split('/')[-3:]))
    
    

with open("new_paths_criann.txt","w") as f:
    for file in new_paths:
        f.write(file)