
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 07:58:01 2020

@author: antoine
"""
import torch
from train import *

def test(model,device,batch_size,test_path):
    
    img_size=0
    dataset = LoadImagesAndLabels(test_path,
                                  img_size,
                                  batch_size,
                                  rect=False)  # rectangular training
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=8,
                            shuffle=False,  # Shuffle=True unless rectangular training is used
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)
    
    model.eval()
    with torch.no_grad():
        rel_err=[]
        for i, (imgs, targets, paths, _,_) in enumerate(dataloader):
            if len(targets)==0:
                continue
            
            input_targets=targets.numpy()
            
            #xywh€[0,1] to xyxy in pixels
            input_targets[:, [2+1, 4+1]] *= imgs.shape[2]
            input_targets[:, [1+1, 3+1]] *= imgs.shape[3]
            x_centers=input_targets[:,2].copy()
            y_centers=input_targets[:,3].copy()
            w=input_targets[:,4].copy()
            h=input_targets[:,5].copy()
            input_targets[:,2]=x_centers-w/2
            input_targets[:,4]=x_centers+w/2
            input_targets[:,3]=y_centers-h/2
            input_targets[:,5]=y_centers+h/2
            
            targets=targets.to(device).half()
            imgs = imgs.to(device).half()
            with torch.cuda.amp.autocast():
                pred=model(imgs,input_targets[:,np.array([0, 2, 3,4,5 ])])
                rel_err_=compute_rel_err(pred,targets)
            for value in rel_err_:
                rel_err.append(value.float())
        
        rel_err=torch.mean(torch.tensor(rel_err))
    return rel_err