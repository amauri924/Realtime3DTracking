
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
from math import cos,sin
import sys
import math as m


def mergeDict(new_dict, old_dict):
   ''' Merge dictionaries and keep values of common keys in list'''
   for key, values in new_dict.items():
       for value in values:
           try:
               old_dict[key].append(value)
           except:
               old_dict[key] = [value]
   return old_dict

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def angle_between_bisse(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1 = v1[:2]
    v2 = v2[:2]
    angle=-np.math.atan2(np.linalg.det([v1,v2]),np.dot(v1,v2))
    if angle <0:
        angle=angle+2*np.pi
    return angle

def get_theta(model_pos, model_rot, view_matrix):
    model_matrix = construct_model_matrix(model_pos, model_rot)
    model2cam=view_matrix@model_matrix
    
    R=model2cam[:3,:3]
    
    tol = sys.float_info.epsilon * 10
  
    if abs(R.item(0,0))< tol and abs(R.item(1,0)) < tol:
       eul1 = 0
       eul2 = m.atan2(-R.item(2,0), R.item(0,0))
       eul3 = m.atan2(-R.item(1,2), R.item(1,1))
    else:   
       eul1 = m.atan2(R.item(1,0),R.item(0,0))
       sp = m.sin(eul1)
       cp = m.cos(eul1)
       eul2 = m.atan2(-R.item(2,0),cp*R.item(0,0)+sp*R.item(1,0))
       eul3 = m.atan2(sp*R.item(0,2)-cp*R.item(1,2),cp*R.item(1,1)-sp*R.item(0,1))
    
    theta=m.asin(R[0,2])*180/np.pi
    to_deg=eul2*180/np.pi
    return theta,to_deg

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

def pixels_to_world(pos, view_matrix, proj_matrix, width, height):
    pixels_homo = np.array([pos[0], pos[1], pos[2], 1])
    return homo_pixels_coords_to_world(pixels_homo, view_matrix, proj_matrix, width, height)


def homo_pixels_coords_to_world(pixels_homo, view_matrix, proj_matrix, width, height):
    to_pixel_matrix = np.array([
        [width/2, 0, 0, width/2],
        [0, -height/2, 0, height/2],
        [0, 0   ,       0,   1]
    ])
    world_to_pixels_matrix=to_pixel_matrix@proj_matrix@view_matrix
    world_to_pixels_matrix=np.vstack((world_to_pixels_matrix, [0,0,0,1]))
    world_coord = np.linalg.inv(world_to_pixels_matrix) @ pixels_homo
    return world_coord

def homo_pixels_coords_to_cam(pixels_homo, view_matrix, proj_matrix, width, height):
    to_pixel_matrix = np.array([
        [width/2, 0, 0, width/2],
        [0, -height/2, 0, height/2],
        [0, 0   ,       0,   1]
    ])
    reor_matrix=np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    cam_to_pixels_matrix=to_pixel_matrix@proj_matrix
    cam_to_pixels_matrix=np.vstack((cam_to_pixels_matrix, [0,0,0,1]))
    pixels_to_cam=reor_matrix@np.linalg.inv(cam_to_pixels_matrix)
    cam_to_pixels_matrix_new=np.linalg.inv(pixels_to_cam)
    cam_coord = pixels_to_cam @ pixels_homo
    return cam_coord



def model_coords_to_pixel(model_pos, model_rot, pos, view_matrix, proj_matrix, width, height):
    point_homo = np.array([pos[0], pos[1], pos[2], 1])
    model_matrix = construct_model_matrix(model_pos, model_rot)
    # print('model_matrix\n', model_matrix)
    world_point_homo = model_matrix @ point_homo
    return homo_world_coords_to_pixel(world_point_homo, view_matrix, proj_matrix, width, height)


def model_coords_to_cam_coords(model_pos, model_rot, pos, view_matrix, proj_matrix, width, height):
    point_homo = np.array([pos[0], pos[1], pos[2], 1])
    model_matrix = construct_model_matrix(model_pos, model_rot)
    model2cam=view_matrix@model_matrix
    cam_point_homo = model2cam @ point_homo
    return cam_point_homo

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
        
        shape_dict={}

            
            
        out_txt=os.path.join(out_dir,seq_name,cam_folder,base_name+".txt")
        out_img=os.path.join(out_dir,seq_name,cam_folder,base_name+".jpg")
        out_npy=os.path.join(out_dir,seq_name,cam_folder,base_name+".npy")
        
    
        with open("test.txt",'a') as f:
            f.write(out_img+'\n')
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
        
        to_pixel_matrix = np.array([
            [width/2, 0, 0, width/2],
            [0, -height/2, 0, height/2],
            [0, 0   ,       0,   1]
        ])
        reor_matrix=np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        cam_to_pixels_matrix=to_pixel_matrix@proj_matrix
        cam_to_pixels_matrix=np.vstack((cam_to_pixels_matrix, [0,0,0,1]))
        pixels_to_cam=reor_matrix@np.linalg.inv(cam_to_pixels_matrix)
        cam_to_pixels_matrix=np.linalg.inv(pixels_to_cam)
        np.save(out_npy,cam_to_pixels_matrix)

        for row in visible_objects:
            obj_class = row["type"] +'.' + row["class"]
            if obj_class not in list_classes:
                list_classes.append(obj_class)
            pos = np.array(row['pos'])
            rot = np.array(row['rot'])
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
            
            class_idx=list_classes.index(obj_class)
            x_min, x_max, y_min, y_max, z_min, z_max = model_size
            width_object=x_max-x_min
            lenght_object=y_max-y_min
            height_object=z_max-z_min
            
            
            points_3dbbox = np.array([
                [0, y_max, 0],
                [0, y_min, 0],
                [0, 0, 0]
            ])


            # projecting cuboid to 2d
            bbox_2d = np.zeros((8, 2))
            pos_3d=np.zeros((8, 2))
            for i, point in enumerate(points_3dbbox):
                pixel_pos = model_coords_to_pixel(pos, rot, point, view_matrix, proj_matrix, width, height)
                bbox_2d[i, :] = pixel_pos[:2]/pixel_pos[-1]
                cam_pos=model_coords_to_cam_coords(pos, rot, point, view_matrix, proj_matrix, width, height)
                cam_pos[1]*=-1
                cam_pos[2]*=-1
                
                if i == 0:
                    p2=np.array([cam_pos[0], cam_pos[2],0])
                elif i == 1:
                    p1=np.array([cam_pos[0], cam_pos[2],0])

            
            v1=p2-p1
            v2=np.array([1, 0,0])
#                theta_bis=angle_between(v2, v1)*180/np.pi
            theta_bisse=angle_between_bisse(v2, v1)
            alpha=theta_bisse-np.math.atan2(cam_pos[0],cam_pos[2])
            if alpha > 2*np.pi:
                print("alpha > 2pi")
                alpha+=-2*np.pi
            
            try:
                shape_dict[str(obj_class)].append((width_object,height_object,lenght_object))
            except:
                shape_dict[str(obj_class)]=[(width_object,height_object,lenght_object)]
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
            
            if center_pixel_pos[0]/width > 1 or center_pixel_pos[1]/height > 1:
                print("error in " + rgb_path)
            
            out_str=str(class_idx)+' '+str(xc)+' '+str(yc)+' '+str(bbox_width)+' '+str(bbox_height)+' '+str(center_pixel_pos[0]/width)+' '+str(center_pixel_pos[1]/height)+' '+str(dist/max_distance)+' '+str(width_object/max_distance)+' '+str(height_object/max_distance)+' '+str(lenght_object/max_distance)+' '+str(theta_bisse/(2*np.pi))+' '+str(alpha/(2*np.pi))+'\n'
            
        
            
            with open(out_txt,"a") as f:
                f.writelines(out_str)
        return shape_dict


root_dir="/save/2020010/amauri03/GTA_Dataset/"
out_dir="/save/2020010/amauri03/GTA_Preprocessed_v4/"
proc=processor(out_dir)
seq_list=[os.path.join(root_dir,seq) for seq in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,seq))]
manager=multiprocessing.Manager()
list_classes=manager.list()
#shape_dict_global=manager.dict()
shape_dict_global={}
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
#                tmp=file
                shape_dict_global=mergeDict(file, shape_dict_global)
        
        
#        for rgb_path in base_name_list:
#            json_path=os.path.join('/',*rgb_path.split('/')[:-1],"unoccluded",rgb_path.split('/')[-1].split('.')[0]+'.json')
#            seq_name=rgb_path.split('/')[-3]
#            cam_folder=rgb_path.split('/')[-2]
#            base_name=rgb_path.split('/')[-1].split('.')[0]
#            
#            shape_dict={}
#    
#                
#                
#            out_txt=os.path.join(out_dir,seq_name,cam_folder,base_name+".txt")
#            out_img=os.path.join(out_dir,seq_name,cam_folder,base_name+".jpg")
#            out_npy=os.path.join(out_dir,seq_name,cam_folder,base_name+".npy")
#            
#        
#            with open("test.txt",'a') as f:
#                f.write(out_img+'\n')
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
#            to_pixel_matrix = np.array([
#                [width/2, 0, 0, width/2],
#                [0, -height/2, 0, height/2],
#                [0, 0   ,       0,   1]
#            ])
#            reor_matrix=np.array([
#                [1, 0, 0, 0],
#                [0, -1, 0, 0],
#                [0, 0, -1, 0],
#                [0, 0, 0, 1]
#            ])
#            cam_to_pixels_matrix=to_pixel_matrix@proj_matrix
#            cam_to_pixels_matrix=np.vstack((cam_to_pixels_matrix, [0,0,0,1]))
#            pixels_to_cam=reor_matrix@np.linalg.inv(cam_to_pixels_matrix)
#            cam_to_pixels_matrix=np.linalg.inv(pixels_to_cam)
#            np.save(out_npy,cam_to_pixels_matrix)
#            
##            rgb = np.array(Image.open(rgb_path))
##            fig, (ax1, ax2) = plt.subplots(2)
##            plt.title('RGB')
##            ax1.imshow(rgb)
#            
#            for row in visible_objects:
#                obj_class = row["type"] +'.' + row["class"]
#                if obj_class not in list_classes:
#                    list_classes.append(obj_class)
#                pos = np.array(row['pos'])
#                rot = np.array(row['rot'])
#                #Get the position of the pixel coords
#                center_pixel_pos = world_coords_to_pixel(pos, view_matrix, proj_matrix, width, height)
#                
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
#                
#                class_idx=list_classes.index(obj_class)
#                x_min, x_max, y_min, y_max, z_min, z_max = model_size
#                width_object=x_max-x_min
#                lenght_object=y_max-y_min
#                height_object=z_max-z_min
#                
#                
#                points_3dbbox = np.array([
#                    [0, y_max, 0],
#                    [0, y_min, 0],
#                    [0, 0, 0]
#                ])
#
#    
#                # projecting cuboid to 2d
#                bbox_2d = np.zeros((8, 2))
#                pos_3d=np.zeros((8, 2))
#                for i, point in enumerate(points_3dbbox):
#                    # point += pos
#                    pixel_pos = model_coords_to_pixel(pos, rot, point, view_matrix, proj_matrix, width, height)
#                    bbox_2d[i, :] = pixel_pos[:2]/pixel_pos[-1]
#                    cam_pos=model_coords_to_cam_coords(pos, rot, point, view_matrix, proj_matrix, width, height)
#                    cam_pos[1]*=-1
#                    cam_pos[2]*=-1
#                    
#                    if i == 0:
##                        ax2.scatter(cam_pos[0], cam_pos[2],c='r')
##                        ax1.scatter(int(bbox_2d[i, 0]), int(bbox_2d[i, 1]),c='r')
#                        p2=np.array([cam_pos[0], cam_pos[2],0])
#                    elif i == 1:
###                        
##                        ax2.scatter(cam_pos[0], cam_pos[2],c='g')
##                        ax1.scatter(int(bbox_2d[i, 0]), int(bbox_2d[i, 1]),c='g')
#                        p1=np.array([cam_pos[0], cam_pos[2],0])
##                    else:
###                        
##                        ax2.scatter(cam_pos[0], cam_pos[2],c='b')
##                        ax1.scatter(int(bbox_2d[i, 0]), int(bbox_2d[i, 1]),c='b')
#                
#                v1=p2-p1
#                v2=np.array([1, 0,0])
##                theta_bis=angle_between(v2, v1)*180/np.pi
#                theta_bisse=angle_between_bisse(v2, v1)
#                alpha=theta_bisse-np.math.atan2(cam_pos[0],cam_pos[2])
#                if alpha > 2*np.pi:
#                    print("alpha > 2pi")
#                    alpha+=-2*np.pi
#                
#                try:
#                    shape_dict[str(obj_class)].append((width_object,height_object,lenght_object))
#                except:
#                    shape_dict[str(obj_class)]=[(width_object,height_object,lenght_object)]
#                    
#                    
##                theta,eul2=get_theta(pos, rot, view_matrix)
#                
##                ax2.text(cam_pos[0], cam_pos[2], "theta_bis:"+str(int(theta_bis)) +"  theta_bisse:"+str(int(theta_bisse))+ "  alpha:"+str(int(alpha))+ "  eul2:"+str(int(eul2)), fontsize=1)
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
#                
#                if center_pixel_pos[0]/width > 1 or center_pixel_pos[1]/height > 1:
#                    print("error in " + rgb_path)
#                
#                out_str=str(class_idx)+' '+str(xc)+' '+str(yc)+' '+str(bbox_width)+' '+str(bbox_height)+' '+str(center_pixel_pos[0]/width)+' '+str(center_pixel_pos[1]/height)+' '+str(dist/max_distance)+' '+str(width_object/max_distance)+' '+str(height_object/max_distance)+' '+str(lenght_object/max_distance)+' '+str(theta_bisse/(2*np.pi))+' '+str(alpha/(2*np.pi))+'\n'
#                
#            
#                
#                with open(out_txt,"a") as f:
#                    f.writelines(out_str)
#            #plot line
#            
##            y_plot= [0,100]
##            x_plot=[0,0]
##            ax2.plot(x_plot,y_plot)
##            ax2.set_xlim([-50,50])
##            
##            plt.show()
##            fig.savefig("fig.png",dpi=800)
#            shape_dict_global=mergeDict(shape_dict, shape_dict_global)
with open("list_classes.txt","w") as f:
    for obj in list_classes:
        f.write(obj+'\n')

with open("list_shapes.json","w") as f:
    f.write(json.dumps(shape_dict_global))

