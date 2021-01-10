# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 10:20:34 2021

@author: antoine
"""
import os
import numpy as np
from math import cos,sin
import shutil
import cv2
import json

def loadKittiFiles (frame) :
  '''
  Load KITTI image (.png), calibration (.txt), velodyne (.bin), and label (.txt),  files
  corresponding to a shot.

  Args:
    frame :  name of the shot , which will be appended to externsions to load
                the appropriate file.
  '''
  
  basedir = '/home/antoine/KITTI/training/' # *nix
  left_cam_rgb= 'image_2'
  label = 'label_2'
  calib = 'calib'
  
  # load image file 
  fn = basedir+ left_cam_rgb + frame+'.png'
  fn = os.path.join(basedir, left_cam_rgb, frame+'.png')
  image_path=fn

  
  # load calibration file
  fn = basedir+ calib + frame+'.txt'
  fn = os.path.join(basedir, calib, frame+'.txt')
  calib_data = {}
  with open (fn, 'r') as f :
    for line in f.readlines():
      if ':' in line :
        key, value = line.split(':', 1)
        calib_data[key] = np.array([float(x) for x in value.split()])
  
  # load label file
  fn = basedir+ label + frame+'.txt'
  fn = os.path.join(basedir, label, frame+'.txt')
  label_data = {}
  with open (fn, 'r') as f :
    for line in f.readlines():
      if len(line) > 3:
        key, value = line.split(' ', 1)
        #print ('key', key, 'value', value)
        if key in label_data.keys() :
          label_data[key].append([float(x) for x in value.split()] )
        else:
          label_data[key] =[[float(x) for x in value.split()]]

  for key in label_data.keys():
    label_data[key] = np.array( label_data[key])
    
  return image_path, label_data, calib_data

def computeBox3D(label, P):

  
    #2D
    left   = label[3]
    top = label[4]
    bbox_width  = label[5]- label[3]
    bbox_height = label[6]- label[4]
    
    centerx_bbox=left+bbox_width/2
    centery_bbox=top+bbox_height/2
      
    #3D
    h = label[7]
    w = label[8]
    l = label[9]
    x = label[10]
    y = label[11]
    z = label[12]
    ry = label[13]
    theta= ry
    
    #get object_center in 3d
    homo_center_3d=np.array([x,y-h/2,z,1]).T
    homo_center_2d=P@homo_center_3d
    
    center_pixel_pos=homo_center_2d[:3]/homo_center_2d[2]
    
    
    if theta <0:
        theta=theta+2*np.pi
      
    alpha=theta-np.math.atan2(x,z)
    
    if alpha <0:
        theta=theta+2*np.pi
    elif alpha >=2*np.pi:
        alpha=alpha-2*np.pi
          
       
    return centerx_bbox,centery_bbox,bbox_width,bbox_height,center_pixel_pos[0],center_pixel_pos[1],z,w,h,l,theta,alpha


def main():
    
    out_dir="/home/antoine/KITTI/processed_training/"
    max_distance=200
    shape_dict={}
    with open("/home/antoine/KITTI/splits/trainval.txt","r") as f:
        frame_list=[frame.split('\n')[0] for frame in f.readlines()]
    
    
    for frame in frame_list:
       left_cam_path, label_data, calib_data = loadKittiFiles(frame)
       
       out_txt=os.path.join(out_dir,frame+".txt")
       out_img=os.path.join(out_dir,frame+".jpg")
       out_npy=os.path.join(out_dir,frame+".npy")
       
       P2_rect = calib_data['P2'].reshape(3,4)
       P2_rect = np.vstack((P2_rect,np.array([0,0,0,1])))
       
       np.save(out_npy,P2_rect)
       shutil.copy(left_cam_path,out_img)
       img=cv2.imread(left_cam_path)
       height,width,_=img.shape
       with open(out_txt,'w') as f:
           for key in label_data.keys ():
               if key == 'Car':
                   obj_id = 0
               elif key == 'Pedestrian':
                   obj_id = 1
               elif key == 'Cyclist':
                   obj_id = 2
               else:
                   continue
               
               for o in range( label_data[key].shape[0]):
                   centerx_bbox,centery_bbox,bbox_width,bbox_height,center_pixel_posx,center_pixel_posy,z,w,h,l,theta,alpha=computeBox3D(label_data[key][o], P2_rect)
                   try:
                       shape_dict[str(obj_id)].append((w,l,h))
                   except:
                       shape_dict[str(obj_id)]=[(w,l,h)]
                   if z<=80:
                       out_str=str(obj_id)+' '+str(centerx_bbox/width)+' '+str(centery_bbox/height)+' '+str(bbox_width/width)+' '+str(bbox_height/height)+' '+str(center_pixel_posx/width)+' '+str(center_pixel_posy/height)+' '+str(z/max_distance)+' '+str(w/max_distance)+' '+str(h/max_distance)+' '+str(l/max_distance)+' '+str(theta/(2*np.pi))+' '+str(alpha/(2*np.pi))+'\n'
                       f.write(out_str)
       
    with open("list_shapes_KITTI.json","w") as f:
        f.write(json.dumps(shape_dict))


if __name__ == '__main__':
    main()