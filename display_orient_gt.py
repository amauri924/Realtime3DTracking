
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:55:03 2020

@author: antoine
"""

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import cv2
from models import *
from utils.datasets import *
from utils.utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt

out_dir="/home/antoine/GT_Alpha/"


def prepare_data_before_forward(imgs,device,targets,augmented_roi,i,nb,epoch,accumulate,multi_scale,img_size_min,img_size_max,img_size,pixel_to_normalized_resized):
    input_targets=augmented_roi.clone()
    

    # Multi-Scale training TODO: short-side to 32-multiple https://github.com/ultralytics/yolov3/issues/358
    # if multi_scale:
    #     if (i + nb * epoch) / accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
    #         img_size = random.choice(range(img_size_min, img_size_max + 1)) * 32
    #         # print('img_size = %g' % img_size)
    #     scale_factor = img_size / max(imgs.shape[-2:])
    #     imgs = F.interpolate(imgs, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        
    
    #xywh€[0,1] to xyxy in pixels
    input_targets[:, [2+1, 4+1]] *= imgs.shape[1]
    input_targets[:, [1+1, 3+1]] *= imgs.shape[2]
    x_centers=input_targets[:,2].clone()
    y_centers=input_targets[:,3].clone()
    w=input_targets[:,4].clone()
    h=input_targets[:,5].clone()
    input_targets[:,2]=x_centers-w/2
    input_targets[:,4]=x_centers+w/2
    input_targets[:,3]=y_centers-h/2
    input_targets[:,5]=y_centers+h/2

    pixel_to_normalized_resized=pixel_to_normalized_resized.to(device)
    resize_matrix=pixel_to_normalized_resized
    resize_matrix[:,0,:]*=imgs.shape[1]
    resize_matrix[:,1,:]*=imgs.shape[2]
    
    
    
    return imgs,targets,input_targets,img_size,resize_matrix


def plot_one_box(x, img, color=None, label=None, line_thickness=2):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)





data_cfg ='data/KITTI.data'
multi_scale=True
img_size=512
if multi_scale:
    img_size_min = round(img_size / 32 / 2)
    img_size_max = round(img_size / 32)
    img_size = img_size_max * 32  # initiate with maximum multi_scale size
else:
    img_size_min=round(img_size / 32)
    img_size_max=round(img_size / 32)

idx_train=''

# Configure run
data_dict = parse_data_cfg(data_cfg)
train_path = data_dict['train']
nc = int(data_dict['classes'])  # number of classes
device=0
epoch=0
batch_size=16

# Dataset
dataset = LoadImagesAndLabels_display(train_path,
                              1280,
                              batch_size,
                              augment=True,
                              rect=False,
                              depth_aug=True)  # rectangular training


train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                            num_replicas=1,
                                                            rank=0
                                                            )


dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=12,
                        shuffle=False,  # Shuffle=True unless rectangular training is used
                        pin_memory=True,
                        collate_fn=dataset.collate_fn,
                        sampler=train_sampler)

nb=len(dataloader)
accumulate=1

alpha_list=[]
for i, (imgs, targets, paths, _,calib,pixel_to_normalized_resized,augmented_roi) in enumerate(tqdm(dataloader)):
    
    if len(targets)>0:
    
        imgs,targets,input_targets,img_size,resize_matrix=prepare_data_before_forward(imgs,device,targets,augmented_roi,i
                                                                                      ,nb,epoch,accumulate,multi_scale,
                                                                                      img_size_min,img_size_max,img_size,pixel_to_normalized_resized)
        idx_targets=input_targets[:,0]
        
        for j in range(len(imgs)):
            img=imgs[j].numpy()
            
            idx_select=(idx_targets== float(j)).nonzero()[:,0]
            
            valid_targets=torch.index_select(input_targets, 0, idx_select)
            
            if len(valid_targets)>0:
                
                
                rois=valid_targets[:,np.array([2, 3,4,5 ])].numpy()
                for k,roi in enumerate(rois):
                    x,y=valid_targets[k,6:8].numpy()* imgs.shape[1]
                    alpha=valid_targets[k,13:14].clone()*360
                    alpha=alpha.item()
                    if alpha<0:
                        alpha+=360
                    alpha_list.append(alpha)
                    # obj_class=int(valid_targets[k,1].item())
                    
                    # if alpha<90 and alpha >= 0:
                    #     angle_dir="90/"
                    # if alpha<180 and alpha >= 90:
                    #     angle_dir="180/"
                    # if alpha<270 and alpha >= 180:
                    #     angle_dir="270/"
                    # if alpha<=360 and alpha >= 270:
                    #     angle_dir="360/"
                    # cv2.circle(img,(round(x),round(y)),5,(0,0,255),3)
                    # roi[roi<0]=0
                    # label=angle_dir + str(obj_class) + '_' + str(round(alpha))+".png"
                    # cropped_img=img[round(roi[1]):round(roi[3]),round(roi[0]):round(roi[2])]
                    # try:
                    #     cv2.imwrite(os.path.join(out_dir,label),cropped_img)
                    # except:
                    #     print("eh")
    
    # if i==100:
    #     break

hist=np.histogram(np.array(alpha_list))

plt.hist(np.array(alpha_list), bins = 100)
plt.show()