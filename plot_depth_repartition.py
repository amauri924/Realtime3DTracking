
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:13:02 2020

@author: antoine
"""

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torch
from models import *
from utils.datasets import *
from utils.utils import *
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt


# Configure run
data_dict = parse_data_cfg('data/3dcent-NS.data')
train_path = data_dict['train']


# Dataset
dataset = LoadImagesAndLabels(train_path,
                              608,
                              1,
                              augment=False,
                              rect=False)  # rectangular training

dataloader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=12,
                        shuffle=True,  # Shuffle=True unless rectangular training is used
                        pin_memory=True,
                        collate_fn=dataset.collate_fn)
depth_list=[]
for i, (imgs, targets, paths, _) in enumerate(dataloader):
    depth=targets[:,8]*200
    if targets.shape[0]>0:
        depth_list.append(depth)
    print(paths)
    
depth_list=np.array(torch.cat(depth_list).tolist()).reshape(-1,1)
plt.figure()
plt.scatter(np.linspace(0,len(depth_list),num=len(depth_list),dtype=np.uint16),depth_list)
plt.show()
#Cluster K-means
model=KMeans(n_clusters=20)
#adapter le modèle de données
model.fit(depth_list)
cluster_centers=model.cluster_centers_
cluster_centers[:,0].sort()