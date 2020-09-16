# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
#from tqdm import tqdm
import time
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torch
import os
from utils.datasets import *
#from net import Depth_Model
import net
from utils.utils import *
#from apex import amp


def compute_loss(pred,target):
    l1_loss=torch.nn.L1Loss()
    depth_target=target[:,8:]
    
    with open("loss_log.txt",'a') as f:
        f.write("pred:"+str(pred)+'\n')
        f.write("target:"+str(depth_target)+'\n\n\n')
    
    loss=l1_loss(pred,depth_target)
    return loss

def compute_rel_err(pred,target):
    depth_target=target[:,8:].clone().detach()*200
    pred_depth=pred.clone().detach()*200
    
    rel_err=abs(depth_target-pred_depth)/depth_target
    
    return rel_err.split(1)
    

def train(data_cfg,img_size,epochs,batch_size=1,accumulate=1):
    
    idx_train=str(batch_size)+'.'+str(accumulate)
    log_path='log_'+idx_train+'.txt'
    result_path='result_'+idx_train+'.txt'
    weights = 'weights' + os.sep
    latest = weights + 'latest_'+idx_train+'.pt'
    best = weights + 'best_'+idx_train+'.pt'
    device = torch_utils.select_device()
    multi_scale = True
    
    if multi_scale:
        img_size_min = round(img_size / 32 / 1.5)
        img_size_max = round(img_size / 32 * 1.5)
        img_size = img_size_max * 32  # initiate with maximum multi_scale size
        
    #Input file path
    data_dict=parse_data_cfg(data_cfg)
    train_path = data_dict['train']
    
    start_epoch=0
    
    #Create model and optimizer
    model=net.model().to(device)
    
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),
                           lr=1e-3,  weight_decay=1e-2)
 
#    model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
    scaler = torch.cuda.amp.GradScaler() 
    
    
    #Create LR scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True,min_lr=1e-7)
    scheduler.last_epoch = start_epoch - 1
    
    #Create dataloader
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  rect=False)  # rectangular training
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=16,
                            shuffle=False,  # Shuffle=True unless rectangular training is used
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)
    
    # Start training
    nb = len(dataloader)
    for epoch in range(start_epoch, epochs):
        epoch_loss=[]
        rel_err=[]
#        depth_model.train()
        model.train()
        for i, (imgs, targets, paths, _,calib) in enumerate(dataloader):
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
            with torch.cuda.amp.autocast(): 
                loss=compute_loss(pred,targets)
            loss_print=loss.clone().detach()
            with open("log.txt",'a') as f:
                f.write(str(i)+': '+str(loss_print)+'\n')
            epoch_loss.append(loss_print.item())
            scaler.scale(loss).backward() 
            
#            with amp.scale_loss(loss, optimizer) as scaled_loss:
#                scaled_loss.backward()
            
            if (i + 1) % accumulate == 0 or (i + 1) == nb:
                scaler.step(optimizer)
                optimizer.zero_grad()
            scaler.update()
            
        rel_err=torch.mean(torch.tensor(rel_err))
        epoch_loss=np.mean(np.array(epoch_loss))
        print("Epoch loss:"+str(epoch_loss))
        print("Relative error:"+str(rel_err))
        scheduler.step(epoch_loss)
        
        
        
        
train('data/GTA_3dcent.data',416,500)