
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
        seq_name=rgb_path.split('/')[-3]
        cam_folder=rgb_path.split('/')[-2]
        base_name=rgb_path.split('/')[-1].split('.')[0]
        

            
            
        out_txt=os.path.join(self.out_dir,seq_name,cam_folder,base_name+".txt")
        out_img=os.path.join(self.out_dir,seq_name,cam_folder,base_name+".jpg")
        out_npy=os.path.join(self.out_dir,seq_name,cam_folder,base_name+".npy")
        
    
    #    with open("test.txt",'a') as f:
    #        f.write(out_img+'\n')
        with open(json_path,"r") as f:
            data=json.load(f)
    
        if os.path.exists(out_txt):
            os.remove(out_txt)
        view_matrix = np.array(data['view_matrix'])
        proj_matrix = np.array(data['proj_matrix'])
        width = data['width']
        height = data['height']
    
        
        visible_objects=data["unoccluded_targets"]
        
    
        shutil.copy(rgb_path,out_img)
        
        to_pixel_matrix=np.array([
            [width/2, 0, 0, width/2],
            [0, height/2, 0, height/2],
            [0, 0   ,       0,   1]])
        
        cam_to_pixel=to_pixel_matrix@proj_matrix
        cam_to_pixel[:,2]*=-1
        
        cam_to_pixel=np.vstack((cam_to_pixel, [0,0,0,1]))
        pixel_to_cam=np.linalg.inv(cam_to_pixel)
        np.save(out_npy,pixel_to_cam)
        
        for row in visible_objects:
            obj_class = row["type"] +'.' + row["class"]
            if obj_class not in list_classes:
                list_classes.append(obj_class)
            pos = np.array(row['pos'])
            
            #Get the position of the pixel coords
            center_pixel_pos = world_coords_to_pixel(pos, view_matrix, proj_matrix, width, height)
            #Z axis is still in m
            dist=center_pixel_pos[-1]
            if dist>max_distance:
                continue
            #Now it is in pixels
            center_pixel_pos/=center_pixel_pos[-1]
            center_pixel_pos=center_pixel_pos[:-1]
    
            #Get model size relative to its center, the values correspond to axis offset relative to 3D center
            #Values are normalized
            model_size=row["model_sizes"]
            x_min=model_size[0]/200
            x_max=model_size[1]/200
            y_min=model_size[2]/200
            y_max=model_size[3]/200
            z_min=model_size[4]/200
            z_max=model_size[5]/200
            
            #Get BBox coords ready for YOLO (x,y,width,height)
            bbox = np.array(row['bbox'])
            bbox[:, 0] *= width
            bbox[:, 1] *= height
            bbox_width, bbox_height = bbox[0, :] - bbox[1, :]
            x,y=bbox[1, :]
            
            xc=(x+bbox_width/2)/width
            yc=(y+bbox_height/2)/height
            

            
            bbox_width*=1/width
            bbox_height*=1/height
            class_idx=list_classes.index(obj_class)
            out_str=str(class_idx)+' '+str(xc)+' '+str(yc)+' '+str(bbox_width)+' '+str(bbox_height)+' '+str(center_pixel_pos[0]/width)+' '+str(center_pixel_pos[1]/height)+' '+str(dist/max_distance)+' '+str(x_min)+' '+str(x_max)+' '+str(y_min)+' '+str(y_max)+' '+str(z_min)+' '+str(z_max)+'\n'
            
            
            
            with open(out_txt,"a") as f:
                f.writelines(out_str)
        return 1


root_dir="/save/2020010/amauri03/GTA_Dataset/"
out_dir="/save/2020010/amauri03/GTA_Preprocessed_v3"
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
        
        
        
        if not os.path.exists(os.path.join(out_dir,seq_name)):
            os.mkdir(os.path.join(out_dir,seq_name))
            
        if not os.path.exists(os.path.join(out_dir,seq_name,cam_folder)):
            os.mkdir(os.path.join(out_dir,seq_name,cam_folder))
        
        
        with multiprocessing.Pool(maxtasksperchild=500) as pool:
            for file in pool.imap(proc,base_name_list):
                tmp=file
        
        
#        for rgb_path in base_name_list:
#            json_path=os.path.join('/',*rgb_path.split('/')[:-1],"unoccluded",rgb_path.split('/')[-1].split('.')[0]+'.json')
#            seq_name=rgb_path.split('/')[-3]
#            cam_folder=rgb_path.split('/')[-2]
#            base_name=rgb_path.split('/')[-1].split('.')[0]
#            
#    
#                
#                
#            out_txt=os.path.join(out_dir,seq_name,cam_folder,base_name+".txt")
#            out_img=os.path.join(out_dir,seq_name,cam_folder,base_name+".jpg")
#            out_npy=os.path.join(out_dir,seq_name,cam_folder,base_name+".npy")
#            
        
        #    with open("test.txt",'a') as f:
        #        f.write(out_img+'\n')
#            with open(json_path,"r") as f:
#                data=json.load(f)
#        
#            if os.path.exists(out_txt):
#                os.remove(out_txt)
#            view_matrix = np.array(data['view_matrix'])
#            proj_matrix = np.array(data['proj_matrix'])
#            width = data['width']
#            height = data['height']
#        
#            
#            visible_objects=data["unoccluded_targets"]
#            
#        
#            shutil.copy(rgb_path,out_img)
#            
#            to_pixel_matrix=np.array([
#                [width/2, 0, 0, width/2],
#                [0, height/2, 0, height/2],
#                [0, 0   ,       0,   1]])
#            
#            cam_to_pixel=to_pixel_matrix@proj_matrix
#            cam_to_pixel[:,2]*=-1
#            
#            cam_to_pixel=np.vstack((cam_to_pixel, [0,0,0,1]))
#            pixel_to_cam=np.linalg.inv(cam_to_pixel)
#            np.save(out_npy,pixel_to_cam)
#            
#            for row in visible_objects:
#                obj_class = row["type"] +'.' + row["class"]
#                if obj_class not in list_classes:
#                    list_classes.append(obj_class)
#                pos = np.array(row['pos'])
#                
#                #Get the position of the pixel coords
#                center_pixel_pos = world_coords_to_pixel(pos, view_matrix, proj_matrix, width, height)
#                #Z axis is still in m
#                dist=center_pixel_pos[-1]
#                if dist>max_distance:
#                    continue
#                #Now it is in pixels
#                center_pixel_pos/=center_pixel_pos[-1]
#                center_pixel_pos=center_pixel_pos[:-1]
#        
#                #Get model size relative to its center, the values correspond to axis offset relative to 3D center
#                #Values are normalized
#                model_size=row["model_sizes"]
#                x_min=model_size[0]/200
#                x_max=model_size[1]/200
#                y_min=model_size[2]/200
#                y_max=model_size[3]/200
#                z_min=model_size[4]/200
#                z_max=model_size[5]/200
#                
#                #Get BBox coords ready for YOLO (x,y,width,height)
#                bbox = np.array(row['bbox'])
#                bbox[:, 0] *= width
#                bbox[:, 1] *= height
#                bbox_width, bbox_height = bbox[0, :] - bbox[1, :]
#                x,y=bbox[1, :]
#                
#                xc=(x+bbox_width/2)/width
#                yc=(y+bbox_height/2)/height
#                
#
#                
#                bbox_width*=1/width
#                bbox_height*=1/height
#                class_idx=list_classes.index(obj_class)
#                out_str=str(class_idx)+' '+str(xc)+' '+str(yc)+' '+str(bbox_width)+' '+str(bbox_height)+' '+str(center_pixel_pos[0]/width)+' '+str(center_pixel_pos[1]/height)+' '+str(dist/max_distance)+' '+str(x_min)+' '+str(x_max)+' '+str(y_min)+' '+str(y_max)+' '+str(z_min)+' '+str(z_max)+'\n'
#                
#                
#                
#                with open(out_txt,"a") as f:
#                    f.writelines(out_str)
with open("list_classes.txt","w") as f:
    for obj in list_classes:
        f.write(obj+'\n')
