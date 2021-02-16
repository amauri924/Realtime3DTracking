
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:33:35 2020

@author: antoine
"""

import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, default="YoloV3_Annotation_Tool-master/Images/", help='Directory where the images are stored')
parser.add_argument('--output_folder', type=str, default='data/3dcent/', help='output path')
args = parser.parse_args()



image_list=[os.path.join(args.input_folder,image) for image in os.listdir(args.input_folder) if not image.endswith('.txt')]
image_list.sort()


train_list=[]
test_list=[]
for i,image_path in enumerate(image_list):
    if i%10<9:
        train_list.append(image_path+'\n')
    else:
        test_list.append(image_path+'\n')
        

with open(os.path.join(args.output_folder,"train.txt"),'w') as f:
    f.writelines(train_list)
    

with open(os.path.join(args.output_folder,"test.txt"),'w') as f:
    f.writelines(test_list)