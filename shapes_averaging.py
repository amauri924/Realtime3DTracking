
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:23:03 2020

@author: antoine
"""

import os
import json
import numpy as np

with open('list_shapes.json','r') as f:
    data = json.load(f)
    
with open("data/coco.names","r") as f:
    classes=[obj.split('\n')[0] for obj in f.readlines()]

avg_shape={}
for class_obj in classes:
    avg_shape[classes.index(class_obj)]=(0,0,0)
    
    
    
for obj_class in data:
    avg_width,avg_height,avg_length=[],[],[]
    for obj in data[obj_class]:
        avg_width.append(obj[0])
        avg_height.append(obj[2])
        avg_length.append(obj[1])
    
    avg_width=np.mean(np.array(avg_width))
    avg_height=np.mean(np.array(avg_height))
    avg_length=np.mean(np.array(avg_length))
    
    avg_shape[classes.index(obj_class)]=(avg_width,avg_height,avg_length)
    

with open("data/3dcent-NS/avg_shapes.json","w") as f:
    f.write(json.dumps(avg_shape))