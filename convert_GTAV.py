
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:56:32 2020

@author: antoine
"""
import os
import json
import numpy as np
import shutil
from PIL import Image
from matplotlib import pyplot as plt

def homo_world_coords_to_pixel(point_homo, view_matrix, proj_matrix, width, height):
    viewed = view_matrix @ point_homo
    projected = proj_matrix @ viewed
    projected /= projected[3]
    to_pixel_matrix = np.array([
        [width/2, 0, 0, width/2],
        [0, -height/2, 0, height/2],
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



seq_dir="/media/antoine/Save_HDD/GTA_Dataset/seq_00004/0/"
out_dir="/media/antoine/Disque dur portable/Detection Methods/Yolov3 Pytorch new/yolov3-master/YoloV3_Annotation_Tool-master/Image_3d/"

base_name_list=[base_name.split(".")[0] for base_name in os.listdir(seq_dir) if base_name.endswith(".jpg")]

list_classes=[]

for base_name in base_name_list:
    rgb_path=os.path.join(seq_dir,base_name+".jpg")
    json_path=os.path.join(seq_dir,"unoccluded",base_name+".json")
    
    
    out_txt=os.path.join(out_dir,base_name+".txt")
    out_img=os.path.join(out_dir,base_name+".jpg")
    with open(json_path,"r") as f:
        data=json.load(f)

    if os.path.exists(out_txt):
        os.remove(out_txt)
    view_matrix = np.array(data['view_matrix'])
    proj_matrix = np.array(data['proj_matrix'])
    width = data['width']
    height = data['height']

    
    visible_objects=data["unoccluded_targets"]
    
    if len(visible_objects)>0:
        shutil.copy(rgb_path,out_img)
    
    
#    rgb = np.array(Image.open(rgb_path))
#    fig = plt.figure(figsize=(16, 16))
#    plt.title('RGB')
#    plt.imshow(rgb)
    
    for row in visible_objects:
        obj_class = row["type"]
        if obj_class not in list_classes:
            list_classes.append(obj_class)
        pos = np.array(row['pos'])
        center_pixel_pos = world_coords_to_pixel(pos, view_matrix, proj_matrix, width, height)
        
#        plt.scatter(center_pixel_pos[0], center_pixel_pos[1])
        
        
        bbox = np.array(row['bbox'])
        bbox[:, 0] *= width
        bbox[:, 1] *= height
        bbox_width, bbox_height = bbox[0, :] - bbox[1, :]
        x,y=bbox[1, :]
        
        xc=(x+bbox_width/2)/width
        yc=(y+bbox_height/2)/height
        
#        plt.scatter(xc*width, yc*height)
        
        bbox_width*=1/width
        bbox_height*=1/width
        class_idx=list_classes.index(obj_class)
        out_str=str(class_idx)+' '+str(xc)+' '+str(yc)+' '+str(bbox_width)+' '+str(bbox_height)+' '+str(center_pixel_pos[0]/width)+' '+str(center_pixel_pos[1]/height)+'\n'
        
        with open(out_txt,"a") as f:
            f.writelines(out_str)
#    plt.show()