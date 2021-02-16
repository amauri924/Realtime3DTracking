
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:56:32 2020

@author: antoine
"""

from tqdm import tqdm
import os
import json
import numpy as np
import shutil
from PIL import Image
from matplotlib import pyplot as plt
import multiprocessing


def homo_world_coords_to_pixel(point_homo, view_matrix, proj_matrix, width, height):
    viewed = view_matrix @ point_homo
    projected = proj_matrix @ viewed
#    projected /= projected[3]
    to_pixel_matrix = np.array([
        [width/2, 0, 0, width/2],
        [0, -height/2, 0, height/2],
        [0, 0   ,       0,   1]
    ])
    in_pixels = to_pixel_matrix @ projected
    return in_pixels


def world_coords_to_pixel(pos, view_matrix, proj_matrix, width, height):
    point_homo = np.array([pos[0], pos[1], pos[2], 1])
    return homo_world_coords_to_pixel(point_homo, view_matrix, proj_matrix, width, height)


def model_coords_to_pixel(model_pos, model_rot, pos, view_matrix, proj_matrix, width, height):
    point_homo = np.array([pos[0], pos[1], pos[2], 1])
    model_matrix = construct_model_matrix(model_pos, model_rot)
    # print('model_matrix\n', model_matrix)
    world_point_homo = model_matrix @ point_homo
    return homo_world_coords_to_pixel(world_point_homo, view_matrix, proj_matrix, width, height)


def create_rot_matrix(euler):
    x = np.radians(euler[0])
    y = np.radians(euler[1])
    z = np.radians(euler[2])

    Rx = np.array([
        [1, 0, 0],
        [0, cos(x), -sin(x)],
        [0, sin(x), cos(x)]
    ], dtype=np.float)
    Ry = np.array([
        [cos(y), 0, sin(y)],
        [0, 1, 0],
        [-sin(y), 0, cos(y)]
    ], dtype=np.float)
    Rz = np.array([
        [cos(z), -sin(z), 0],
        [sin(z), cos(z), 0],
        [0, 0, 1]
    ], dtype=np.float)
    result = Rx @ Ry @ Rz
    return result


def construct_model_matrix(position, rotation):
    view_matrix = np.zeros((4, 4))
    # view_matrix[0:3, 3] = camera_pos
    view_matrix[0:3, 0:3] = create_rot_matrix(rotation)
    view_matrix[0:3, 3] = position
    view_matrix[3, 3] = 1

    return view_matrix


class processor:
    def __init__(self,out_dir):
        self.out_dir=out_dir
        
        
    def __call__(self,rgb_path):
        json_path=os.path.join('/',*rgb_path.split('/')[:-1],"unoccluded",rgb_path.split('/')[-1].split('.')[0]+'.json')
        with open(json_path,"r") as f:
            data=json.load(f)
        visible_objects=data["unoccluded_targets"]

        
        for row in visible_objects:
            obj_class = row["type"] +'.' + row["class"]
            if obj_class not in list_classes:
                list_classes.append(obj_class)

        return 1


root_dir="/home/antoine/remote_criann/GTA_Dataset/"
out_dir="/home/antoine/remote_criann/GTA_Preprocessed_v2"
proc=processor(out_dir)
seq_list=[os.path.join(root_dir,seq) for seq in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,seq))]
manager=multiprocessing.Manager()
list_classes=manager.list()

for seq_dir in tqdm(seq_list):
    seq_name=seq_dir.split('/')[-1]
    
    cam_list=[os.path.join(seq_dir,cam) for cam in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir,cam))]
    for cam_dir in cam_list:

        cam_folder=cam_dir.split('/')[-1]
        max_distance=200
        base_name_list=[os.path.join(seq_dir,cam_dir,base_name) for base_name in os.listdir(os.path.join(seq_dir,cam_dir)) if base_name.endswith(".jpg")]

        
        with multiprocessing.Pool(maxtasksperchild=500) as pool:
            for file in pool.imap(proc,base_name_list):
                tmp=file

with open("list_classes.txt","w") as f:
    for obj in list_classes:
        f.write(obj+'\n')
