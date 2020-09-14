
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:20:22 2020

@author: antoine
"""

import torch

sd0=torch.load("/home/antoine/Realtime3DTracking/weights/test_0.pt")['model']
sd1=torch.load("/home/antoine/Realtime3DTracking/weights/test_1.pt")['model']
sd2=torch.load("/home/antoine/Realtime3DTracking/weights/test_2.pt")['model']
sd3=torch.load("/home/antoine/Realtime3DTracking/weights/test_3.pt")['model']



for k in sd1:
    v1 = sd1[k]
    v3 = sd3[k]
    if (v1!=v3).all():
        print(k)
