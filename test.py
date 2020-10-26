
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 07:58:01 2020

@author: antoine
"""
import torch
from train import *
from tqdm import tqdm



def get_scaling_ratio(model,device,batch_size,test_path,img_size):
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
        all_pred=torch.tensor([],device=device)
        all_targets=torch.tensor([],device=device)
        for i, (imgs, targets, paths, _,_) in enumerate(tqdm(dataloader)):
#            t=time.time()
            if len(targets)==0:
                continue
            
            input_targets=targets.clone()
            
            #xywh€[0,1] to xyxy in pixels
            input_targets[:, [2+1, 4+1]] *= imgs.shape[2]
            input_targets[:, [1+1, 3+1]] *= imgs.shape[3]
            x_centers=input_targets[:,2].clone()
            y_centers=input_targets[:,3].clone()
            w=input_targets[:,4].clone()
            h=input_targets[:,5].clone()
            input_targets[:,2]=x_centers-w/2
            input_targets[:,4]=x_centers+w/2
            input_targets[:,3]=y_centers-h/2
            input_targets[:,5]=y_centers+h/2
            
            targets=targets.to(device)
            imgs = imgs.to(device,non_blocking=True)
            input_targets=input_targets.to(device,non_blocking=True)
            with torch.cuda.amp.autocast():
                pred=model(imgs,input_targets[:,np.array([0, 2, 3,4,5 ])])

            all_pred=torch.cat((all_pred,pred*200),0)
            all_targets=torch.cat((all_targets,targets[:,8:]*200),0)
        ratio=torch.median(all_targets)/torch.median(all_pred)
    return ratio.cpu().item()

def test(model,device,batch_size,test_path,img_size,ratio=None):
    
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
        for i, (imgs, targets, paths, _,_) in enumerate(tqdm(dataloader)):
#            t=time.time()
            if len(targets)==0:
                continue
            
            input_targets=targets.clone()
            
            #xywh€[0,1] to xyxy in pixels
            input_targets[:, [2+1, 4+1]] *= imgs.shape[2]
            input_targets[:, [1+1, 3+1]] *= imgs.shape[3]
            x_centers=input_targets[:,2].clone()
            y_centers=input_targets[:,3].clone()
            w=input_targets[:,4].clone()
            h=input_targets[:,5].clone()
            input_targets[:,2]=x_centers-w/2
            input_targets[:,4]=x_centers+w/2
            input_targets[:,3]=y_centers-h/2
            input_targets[:,5]=y_centers+h/2
            
            targets=targets.to(device)
            imgs = imgs.to(device,non_blocking=True)
            input_targets=input_targets.to(device,non_blocking=True)
            with torch.cuda.amp.autocast():
                pred=model(imgs,input_targets[:,np.array([0, 2, 3,4,5 ])])
            if ratio:
                rel_err_=compute_rel_err(pred.float()*ratio,targets.float())
            else:
                rel_err_=compute_rel_err(pred.float(),targets.float())
            for value in rel_err_:
                if torch.isfinite(value).all():
                    rel_err.append(value.float())
            del rel_err_,pred,input_targets,h,w,x_centers,y_centers
            del targets,imgs,paths,i
#            print("FPS:"+str(1/(time.time()-t)))
        rel_err=torch.mean(torch.tensor(rel_err).float())
    return rel_err


if __name__== "__main__":
    device = torch_utils.select_device()
    model=net.model().to(device)

    chkpt = torch.load("/home/antoine/Realtime3DTracking/weights/latest_training/best.pt", map_location=device)  # load checkpoint
    model.load_state_dict(chkpt['model'])

    if chkpt['optimizer'] is not None:
#        optimizer.load_state_dict(chkpt['optimizer'])
        best_fitness = chkpt['best_fitness']

    start_epoch = chkpt['epoch'] + 1
    del chkpt
    
    
    data_dict=parse_data_cfg('data/GTA_3dcent.data')
    test_path= data_dict['test']
    valid_path=data_dict['valid']
    ratio=get_scaling_ratio(model,device,1,test_path,img_size=1280)
    rel_err=test(model,device,1,valid_path,img_size=1280,ratio=ratio)
    print("relative_error:" + str(rel_err))
    rel_err=test(model,device,1,valid_path,img_size=1280)
    print("relative_error:" + str(rel_err))