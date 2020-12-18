
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:50:26 2020

@author: antoine
"""

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
import cv2
import numpy as np
from pyquaternion import Quaternion
from typing import Tuple, List
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
import os
import matplotlib.pyplot as plt
from PIL import Image
import json


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


def get_sample_data(nusc, sample_data_token: str,
                    box_vis_level: BoxVisibility = BoxVisibility.ANY,
                    selected_anntokens: List[str] = None,
                    use_flat_vehicle_coordinates: bool = False) -> \
        Tuple[str, List[Box], np.array]:
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def get_3d_center(points,view):
    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view
    
    nbr_points = 1
    
    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.array([1])))
    points=points.reshape(4,1)
    
    
    points = np.dot(viewpad, points)
    points = points[:3, :]


    points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
    
    
    return points[:2]
    

def transform_3d_to_2d(points,view):
    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view
    
    nbr_points = 1
    nbr_points = points.shape[1]
    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
#    points=points.reshape(4,1)
    
    
    points = np.dot(viewpad, points)
    points = points[:3, :]


    points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
    
    
    return points[:2]

def xyxy2xywh(bbox):
    x1=bbox[0][0]
    x2=bbox[1][0]
    y1=bbox[0][1]
    y2=bbox[1][1]
    
    w=x2-x1
    h=y2-y1
    new_x=x1+w/2
    new_y=y1+h/2
    
    return new_x,new_y,w,h
    


nusc = NuScenes(version='v1.0-trainval', dataroot='/save/2020010/amauri03/Nuscenes', verbose=True)

scenes=nusc.scene
sample_list=nusc.sample
all_classes=[]
valid_classes=['vehicle','human','animal']
max_distance=200
shape_dict_global={}
list_classes=[]
coco_classes=["person",
"bicycle",
"car",
"motorcycle",
"airplane",
"bus",
"train",
"truck"]


for scene in scenes:

    
    scene_token=scene["token"]
    my_samples=[sample for sample in sample_list if sample["scene_token"]==scene_token]
    scene_id=int(scene['name'].split('-')[-1])

    
    for sample_id,sample in enumerate(my_samples):
        
        labels=[]
        sensor = 'CAM_FRONT'
        cam_front_data = nusc.get('sample_data', sample['data'][sensor])
        
        ego_pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
        cal_sensor = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])

        
        data_path, boxes, camera_intrinsic = get_sample_data(nusc,cam_front_data['token'],
                                                                                   box_vis_level=BoxVisibility.ANY)
        viewpad = np.eye(4)
        viewpad[:camera_intrinsic.shape[0], :camera_intrinsic.shape[1]] = camera_intrinsic
        
        np.save("/save/2020010/amauri03/NuScenes_3d_BBOX/3D_BBOX_data/scene%i_sample%i.npy"%(scene_id,sample_id),viewpad)
        
        # rgb = np.array(Image.open(data_path))
        # fig, (ax1, ax2) = plt.subplots(2)
        # plt.title('RGB')
        # ax1.imshow(rgb)
        # y_plot= [0,100]
        # x_plot=[0,0]
        # ax2.plot(x_plot,y_plot)
        # ax2.set_xlim([-50,50])
        
        visible_boxes=[]
        for box in boxes:
            
            visibility=nusc.get('sample_annotation',box.token)['visibility_token']
            sample_annotation=nusc.get('sample_annotation',box.token)
            # nusc.render_annotation(sample_annotation['token'])
            if int(visibility)>1:
                label=nusc.get('sample_annotation',box.token)['category_name'].split('.')[0]
                if len(nusc.get('sample_annotation',box.token)['category_name'].split('.'))>1 and label in valid_classes:
                
                    obj_class=nusc.get('sample_annotation',box.token)['category_name'].split('.')[1]
                    
                    if label=="human":
                        obj_class="person"
                    
                    elif obj_class == "trailer":
                        obj_class="truck"
                    
                    elif obj_class not in coco_classes:
                        break
                    
                    # if obj_class not in list_classes:
                    #     list_classes.append(obj_class)
                        
                    visible_boxes.append(box)
                    corners=box.corners()
                    
                    # for i in range(corners.shape[1]):
                    #     if i <4:
                    #         ax2.scatter(corners[0,i], corners[2,i],c='r',s=5)
                    #     else:
                    # ax2.scatter(corners[0,0], corners[2,0],c='g',s=1)
                    # ax2.scatter(corners[0,4], corners[2,4],c='r',s=1)
                    

                    p2=np.array([corners[0,0], corners[2,0],0])
                    p1=np.array([corners[0,4], corners[2,4],0])

                    v1=p2-p1
                    v2=np.array([1, 0,0])
                    
                    
                    
                    theta_bisse=angle_between_bisse(v2, v1)
                    alpha=theta_bisse-np.math.atan2(box.center[0],box.center[2])
                    
                    if alpha<0:
                        alpha+=2*np.pi
                    elif alpha>=2*np.pi:
                        alpha+=-2*np.pi
                        
                    
                    # ax2.text(corners[0,0], corners[2,0], "theta_bis:"+str(int(theta_bisse*180/np.pi)), fontsize=1)
                    
                    
                    # ax2.scatter(box.center[0], box.center[2],c='r',s=5)
                    corners=transform_3d_to_2d(corners,camera_intrinsic)
                    
                    center=get_3d_center(box.center,camera_intrinsic)
                    # ax1.scatter(center[0], center[1],c='r',s=5)
                    
                    # for i in range(corners.shape[1]):
                    #     if i <4:
                    # ax1.scatter(corners[0,0], corners[1,0],c='g',s=1)
                    # ax1.scatter(corners[0,4], corners[1,4],c='r',s=1)
                    
                    
                    
                    #     else:
                    #         ax1.scatter(corners[0,i], corners[1,i],c='g')
                        
                
                    
                    
                    
                    dist=box.center[2]
                    size_wlh=box.wlh
                    
                    try:
                        shape_dict_global[obj_class].append((size_wlh[0],size_wlh[1],size_wlh[2]))
                    except:
                        shape_dict_global[obj_class]=[(size_wlh[0],size_wlh[1],size_wlh[2])]
                    
                    
                    labels.append((obj_class,[(min(corners[0,:]),min(corners[1,:])),(max(corners[0,:]),max(corners[1,:]))],center,
                                   dist,size_wlh,theta_bisse/(2*np.pi),alpha/(2*np.pi)))
                
        # fig.savefig("fig_%s.png"%str(sample_id),dpi=800)
        
        
        im = cv2.imread(data_path)
        im_width=im.shape[1]
        im_height=im.shape[0]
        cv2.imwrite("/save/2020010/amauri03/NuScenes_3d_BBOX/3D_BBOX_data/scene%i_sample%i.png"%(scene_id,sample_id), im)
        with open("/save/2020010/amauri03/NuScenes_3d_BBOX/3D_BBOX_data/scene%i_sample%i.txt"%(scene_id,sample_id),'w') as f:
            for obj_class,bbox,center,dist,size_wlh,theta_bisse,alpha in labels:
                
                obj_idx=coco_classes.index(obj_class)
                x,y,w,h=xyxy2xywh(bbox)
                center_x=center[0,0]/im_width
                center_y=center[1,0]/im_height
                x/=im_width
                w/=im_width
                y/=im_height
                h/=im_height
                if x>=1 or y>=1 or w>=1 or h>=1 or center_x>=1 or center_y>=1 or x<0 or y<0 or h<0 or w<0 or center_x<0 or center_y<0 or dist>max_distance:
                    continue
                f.write(str(obj_idx)+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+' '+str(center_x)+' '+str(center_y)+' '+str(dist/max_distance)+' '+str(size_wlh[0]/max_distance)+' '+str(size_wlh[2]/max_distance)+' '+str(size_wlh[1]/max_distance)+' '+str(theta_bisse)+' '+str(alpha)+'\n')

        with open("list.txt",'a') as f:
            f.write("/save/2020010/amauri03/NuScenes_3d_BBOX/3D_BBOX_data/scene%i_sample%i.png"%(scene_id,sample_id)+'\n')


with open("list_classes.txt","w") as f:
    for obj in coco_classes:
        f.write(obj+'\n')

with open("list_shapes.json","w") as f:
    f.write(json.dumps(shape_dict_global))





#nusc.render_sample_data(cam_front_data['token'])

