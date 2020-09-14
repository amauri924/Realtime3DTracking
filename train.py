import argparse
import time

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torch
import math
import tqdm
import test as test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
import sys

#sys.path.append("/save/2020010/amauri03/package_perso")

#      0.109      0.297       0.15      0.126       7.04      1.666      4.062     0.1845       42.6       3.34      12.61      8.338     0.2705      0.001         -4        0.9     0.0005   320 giou + best_anchor False
hyp = {'giou': 1.666,  # giou loss gain
       'xy': 4.062,  # xy loss gain
       'wh': 0.1845,  # wh loss gain
       'cls': 42.6,  # cls loss gain
       'cls_pw': 3.34,  # cls BCELoss positive_weight
       'obj': 12.61,  # obj loss gain
       'obj_pw': 8.338,  # obj BCELoss positive_weight
       'iou_t': 0.2705,  # iou target-anchor training threshold
#       'lr0': 0.001,  # initial learning rate
       'lr0': 1e-3,  # initial learning rate
       'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.90,  # SGD momentum
       'weight_decay': 0.0005}  # optimizer weight decay

def print_mutation(hyp, results):
    # Write mutation results
    a = '%11s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%11.4g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%11.3g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if opt.bucket:
        os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt
        with open('evolve.txt', 'a') as f:  # append result
            f.write(c + b + '\n')
        os.system('gsutil cp evolve.txt gs://%s' % opt.bucket)  # upload evolve.txt
    else:
        with open('evolve.txt', 'a') as f:
            f.write(c + b + '\n')

def setup_net_transfert(model):
    model.transfer=True
    model.eval()
    if type(model) is nn.parallel.DistributedDataParallel:
        model.module.depth_pred.train()
        model.module.featurePooling.train()
        
    else:
        model.depth_pred.train()
        model.featurePooling.train()
    return model



def check_updated_grad(previous_state_dict,model):
    list_k=[k for k,v in previous_state_dict.items()]
    previous_model_state={k: v for k,v in previous_state_dict.items()}
    actual_model_state={k: v for k,v in model.state_dict().items()}
    for k in actual_model_state:
        if not torch.all(torch.eq(actual_model_state[k], previous_model_state[k])).cpu().item():
            print("%s updated"%k)

def print_batch_results(mloss,i,loss_items,loss_scheduler,epoch,epochs,optimizer,nb,targets,img_size,log_path):
    # Print batch results
    mloss = (mloss * i + loss_items.cpu()) / (i + 1)  # update mean losses
    loss_scheduler.append(mloss[-1])
    # s = ('%8s%12s' + '%10.3g' * 7) % ('%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, nb - 1), *mloss, len(targets), time.time() - t)
    for x in optimizer.param_groups:
        s = ('%8s%12s' + '%10.3g' * 11) % (
            '%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, nb - 1), *mloss, len(targets), img_size, x['lr'])
    with open(log_path, 'a') as logfile:
        logfile.write(s+"\n")
    return mloss,loss_scheduler,s

def compute_batch_loss(pred,pred_center,depth_pred, targets, model,imgs,calib,opt,resize_matrix):
    # Compute loss
    try:
        loss, loss_items = compute_loss(pred,pred_center,depth_pred, targets, model,imgs.shape[2:],calib,resize_matrix, giou_loss=not opt.xywh)
        return loss,loss_items
    except:
        with open("debug_"+str(opt.run_id)+".txt",'a') as f:
            f.write("error in loss\n")
        return None,None

def prepare_data_before_forward(imgs,device,targets,i,nb,epoch,accumulate,multi_scale,img_size_min,img_size_max,idx_train,paths,img_size,pixel_to_normalized_resized):
    input_targets=targets.numpy()
    targets = targets.to(device)
    

    # Multi-Scale training TODO: short-side to 32-multiple https://github.com/ultralytics/yolov3/issues/358
    if multi_scale:
        if (i + nb * epoch) / accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
            img_size = random.choice(range(img_size_min, img_size_max + 1)) * 32
            # print('img_size = %g' % img_size)
        scale_factor = img_size / max(imgs.shape[-2:])
        imgs = F.interpolate(imgs, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        
    
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

    pixel_to_normalized_resized=pixel_to_normalized_resized.to(device)
    resize_matrix=pixel_to_normalized_resized
    resize_matrix[:,0,:]*=imgs.shape[2]
    resize_matrix[:,1,:]*=imgs.shape[3]
    with open("log_time"+idx_train+str(opt.run_id)+".txt",'a') as f:
        f.write("step %i\n"%i)
        f.write('Paths:\n'+str(paths)+'\n')
    
    return imgs,targets,input_targets,img_size,resize_matrix

def run_test_and_save(model,optimizer,s,data_cfg,batch_size,cfg,log_path,result_path,best_fitness,epoch,epochs,save_freq,latest,best):
    with open(log_path, 'a') as logfile:
        logfile.write("testing \n")
    with torch.no_grad():
        
        if type(model) is nn.parallel.DistributedDataParallel:
            results,_ = test.test(cfg, data_cfg, batch_size=batch_size, img_size=opt.img_size,
                                      model=model.module,
                                      conf_thres=0.1)
            results_training,_ = test.test(cfg, 'data/3dcent-NS-training.data', batch_size=batch_size, img_size=opt.img_size,
                                      model=model.module,
                                      conf_thres=0.1)
        else:
            results,_ = test.test(cfg, data_cfg, batch_size=batch_size, img_size=opt.img_size,
                                      model=model,
                                      conf_thres=0.1)
            results_training,_ = test.test(cfg, 'data/3dcent-NS-training.data', batch_size=batch_size, img_size=opt.img_size,
                                      model=model,
                                      conf_thres=0.1)
    # Write epoch results
    with open(result_path, 'a') as file:
        file.write(s +'|||'+ '%11.3g' * 6 % results + '\n')  # P, R, mAP, F1, center_abs_err, depth_abs_err
    
    with open(result_path.split('.')[0]+"_training.txt", 'a') as file:
        file.write(s +'|||'+ '%11.3g' * 6 % results_training + '\n')  # P, R, mAP, F1, center_abs_err, depth_abs_err
    
    # Update best map
    fitness = results[-1]+(1-results[2])
    if fitness < best_fitness and fitness!=0:
        print("best error replaced by %f"%fitness)
        best_fitness = fitness
    save = ((not opt.nosave) and (epoch%save_freq==0)) or ((not opt.evolve) and (epoch == epochs - 1))
    if save:
        with open(log_path, 'a') as logfile:
            logfile.write("saving...\n")
        with open(result_path, 'r') as file:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                 'best_fitness': best_fitness,
                     'training_results': file.read(),
                 'model': model.module.state_dict() if type(
                     model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                 'optimizer': optimizer.state_dict()}

        # Save latest checkpoint
        torch.save(chkpt, latest)
        if opt.bucket:
            os.system('gsutil cp %s gs://%s' % (latest, opt.bucket))  # upload to bucket
#
        # Save best checkpoint
        if not math.isnan(fitness):
            if best_fitness == fitness:
                torch.save(chkpt, best)
        
    return results,best_fitness


def train(
        cfg,
        data_cfg,
        img_size=416,
        epochs=100,  # 500200 batches at bs 16, 117263 images = 273 epochs
        batch_size=16,
        accumulate=4,  # effective bs = batch_size * accumulate = 8 * 8 = 64
        freeze_backbone=False
):
    idx_train=str(batch_size)+'.'+str(accumulate)
    log_path='log_'+idx_train+str(opt.run_id)+'.txt'
    result_path='result_'+idx_train+str(opt.run_id)+'.txt'
    init_seeds()
    weights = 'weights' + os.sep
    latest = weights + 'latest_'+idx_train+str(opt.run_id)+'.pt'
    best = weights + 'best_'+idx_train+str(opt.run_id)+'.pt'
    device = torch_utils.select_device()
    multi_scale = True

    if multi_scale:
        img_size_min = round(img_size / 32 / 1.5)
        img_size_max = round(img_size / 32 * 1.5)
        img_size = img_size_max * 32  # initiate with maximum multi_scale size

    # Configure run
    data_dict = parse_data_cfg(data_cfg)
    train_path = data_dict['train']
    nc = int(data_dict['classes'])  # number of classes

    # Initialize model
    model = Darknet(cfg,hyp,transfer=True).to(device)
    if opt.transfer:
        for name,param in model.named_parameters():
            if not name.split('.')[0]=="depth_pred" and not name.split('.')[0]=="Yolov3":
                
                param.requires_grad = False
            else:
                if name.split('.')[0]=="Yolov3" and int(name.split('.')[2])>-1:
                    param.requires_grad = False
                else:
                    print(name)
    # Optimizer
#    optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=hyp['lr0'],  weight_decay=hyp['weight_decay'])

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_fitness = 1e6
    if opt.resume or opt.transfer:  # Load previously saved model
        if opt.transfer and not opt.resume:  # Transfer learning
#            nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)
            chkpt = torch.load(weights + 'NuScene.pt', map_location=device)
            model_state=[k for k,v in model.state_dict().items()]
            model.load_state_dict({k: v for k, v in chkpt['model'].items() if k in model_state},strict=False)

#            for p in model.parameters():
#                p.requires_grad = True if p.shape[0] == nf else False

        else:  # resume from latest.pt
            if opt.bucket:
                os.system('gsutil cp gs://%s/latest.pt %s' % (opt.bucket, latest))  # download from bucket
            chkpt = torch.load(latest, map_location=device)  # load checkpoint
            model.load_state_dict(chkpt['model'])

        if chkpt['optimizer'] is not None:
            if opt.resume:
                optimizer.load_state_dict(chkpt['optimizer'])
                best_fitness = chkpt['best_fitness']

        if chkpt['training_results'] is not None:
            with open('results_'+str(opt.run_id)+'.txt', 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt

    else:  # Initialize model with backbone (optional)
        if '-tiny.cfg' in cfg:
            cutoff = load_darknet_weights(model, weights + 'yolov3-tiny.conv.15')
        else:
            cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')

        # Remove old results
        for f in glob.glob('*_batch*.jpg') + glob.glob('results.txt'):
            os.remove(f)

    min_lr=1e-4*hyp['lr0']

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True,min_lr=min_lr)
    scheduler.last_epoch = start_epoch - 1



    # Dataset
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  augment=True,
                                  rect=opt.rect)  # rectangular training

    #Init the model weights
    model.depth_pred.init_weights()

    # Mixed precision training https://github.com/NVIDIA/apex
    mixed_precision = True
    if mixed_precision:
        try:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
            print("using mixed precision")
        except:  # not installed: install help: https://github.com/NVIDIA/apex/issues/259
            mixed_precision = False
            print("using standard precision")

    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        with open(log_path, 'w') as logfile:
            logfile.write("nb GPU : %i\n"%torch.cuda.device_count())
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
    
        model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters = True)


    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=opt.num_workers,
                            shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)
#                            sampler=sampler)



    # Start training
    model_info(model, report='summary')  # 'full' or 'summary'
    nb = len(dataloader)
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    t0 = time.time()
    with open(log_path, 'a') as logfile:
        logfile.write("nb epochs : %i\n"%epochs)
        
    
    for epoch in range(start_epoch, epochs):
        loss_scheduler=[]
        model.train()
        if opt.transfer:
            model=setup_net_transfert(model)

        with open(log_path, 'a') as logfile:
            logfile.write("Epoch: %i"%epoch)

        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True
        mloss = torch.zeros(8)  # mean losses

        for i, (imgs, targets, paths, _,calib,pixel_to_normalized_resized) in enumerate(dataloader):

            imgs = imgs.to(device)
            if opt.depth_aug:
                targets=rois_augmentation_for_depth(targets,0.2,0.02)
            if imgs.shape[0]<torch.cuda.device_count():
                continue
            if imgs.shape[0]!=opt.batch_size:
                continue
            if len(targets)==0:
                continue
        
            imgs,targets,input_targets,img_size,resize_matrix=prepare_data_before_forward(imgs,device,targets,i,nb,epoch,accumulate,
                                        multi_scale,img_size_min,img_size_max,idx_train,paths,img_size,pixel_to_normalized_resized)
            # Run model
            t_pred=time.time()
            pred,pred_center,depth_pred = model(imgs,targets=input_targets)
            t_pred=time.time()-t_pred
            
            with open("log_time"+idx_train+str(opt.run_id)+".txt",'a') as f:
                f.write("tpred %f\n"%t_pred)

            loss,loss_items=compute_batch_loss(pred,pred_center,depth_pred, targets, model,imgs,calib,opt,resize_matrix)
            if loss is None:
                continue
            if torch.isnan(loss):
                with open(log_path, 'a') as logfile:
                    logfile.write('WARNING: nan loss detected, ending training \n')
                return pred


            t_back=time.time()
            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            t_back=time.time()-t_back
            with open("log_time"+idx_train+str(opt.run_id)+".txt",'a') as f:

                f.write("tback %f\n\n"%t_back)

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == nb:
                optimizer.step()
                optimizer.zero_grad()
                
            mloss,loss_scheduler,s=print_batch_results(mloss,i,loss_items,loss_scheduler,epoch,epochs,optimizer,nb,targets,img_size,log_path)
            
#            chkpt = {'epoch': epoch,
#                 'model': model.module.state_dict() if type(
#                     model) is nn.parallel.DistributedDataParallel else model.state_dict(),
#                 'optimizer': optimizer.state_dict()}
#
#            # Save latest checkpoint
#            torch.save(chkpt, weights+"test_%i.pt"%i)
            
            

        loss_scheduler=torch.mean(torch.tensor(loss_scheduler)).item()
        print("Epoch loss:"+str(loss_scheduler))
        scheduler.step(loss_scheduler)
        # Report time
        dt = (time.time() - t0) / 3600
        with open(log_path, 'a') as logfile:
            logfile.write('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, dt))

            
        if hyp['lr0']<min_lr*10:
            save_freq=1
        else:
            save_freq=10
        # Calculate mAP (always test final epoch, skip first 5 if opt.nosave)
        if (not (opt.notest or (opt.nosave and epoch < 10)) and epoch%save_freq==0) or (epoch == epochs - 1):
            results,best_fitness=run_test_and_save(model,optimizer,s,data_cfg,batch_size,cfg,log_path,result_path,best_fitness,epoch,epochs,save_freq,latest,best)

    with open(log_path, 'a') as logfile:
        logfile.write("ending \n")
    return results





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', default='', help='number of epochs')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=3,
                        help='batch size')
    parser.add_argument('--accumulate', type=int, default=1, help='number of batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-3dcent-NS.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/GTA_3dcent.data', help='coco.data file path')
    parser.add_argument('--multi-scale', default=True, help='train at (1/1.5)x - 1.5x sizes')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--rect', default=False, help='rectangular training')
    parser.add_argument('--resume', default=False, help='resume training flag')
    parser.add_argument('--depth_aug', default=False, help='resume training flag')
    parser.add_argument('--transfer', default=False, help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=12, help='number of Pytorch DataLoader workers')
    parser.add_argument('--nosave', default=False, help='only save final checkpoint')
    parser.add_argument('--notest', default=False, help='only test final epoch')
    parser.add_argument('--xywh', action='store_true', help='use xywh loss instead of GIoU loss')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--var', default=0, type=int, help='debug variable')
    opt = parser.parse_args()
    print(opt)

    if opt.evolve:
        opt.notest = True  # only test final epoch
        opt.nosave = True  # only save final checkpoint

    # Train
    results = train(opt.cfg,
                    opt.data_cfg,
                    img_size=opt.img_size,
                    epochs=opt.epochs,
                    batch_size=opt.batch_size,
                    accumulate=opt.accumulate
                    )

#    # Evolve hyperparameters (optional)
#    if opt.evolve:
#        gen = 1000  # generations to evolve
#        print_mutation(hyp, results)  # Write mutation results
#
#        for _ in range(gen):
#            # Get best hyperparameters
#            x = np.loadtxt('evolve.txt', ndmin=2)
#            fitness = x[:, 2] * 0.9 + x[:, 3] * 0.1  # fitness as weighted combination of mAP and F1
#            x = x[fitness.argmax()]  # select best fitness hyps
#            for i, k in enumerate(hyp.keys()):
#                hyp[k] = x[i + 5]
#
#            # Mutate
#            init_seeds(seed=int(time.time()))
#            s = [.15, .15, .15, .15, .15, .15, .15, .15, 0, 0, 0, 0]  # fractional sigmas
#            for i, k in enumerate(hyp.keys()):
#                x = (np.random.randn(1) * s[i] + 1) ** 2.0  # plt.hist(x.ravel(), 300)
#                hyp[k] *= float(x)  # vary by 20% 1sigma
#
#            # Clip to limits
#            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay']
#            limits = [(1e-4, 1e-2), (0, 0.70), (0.70, 0.98), (0, 0.01)]
#            for k, v in zip(keys, limits):
#                hyp[k] = np.clip(hyp[k], v[0], v[1])
#
#            # Train mutation
#            results = train(opt.cfg,
#                            opt.data_cfg,
#                            img_size=opt.img_size,
#                            epochs=opt.epochs,
#                            batch_size=opt.batch_size,
#                            accumulate=opt.accumulate)
#
#            # Write mutation results
#            print_mutation(hyp, results)

#             # Plot results
#             import numpy as np
#             import matplotlib.pyplot as plt
#             a = np.loadtxt('evolve_1000val.txt')
#             x = a[:, 2] * a[:, 3]  # metric = mAP * F1
#             weights = (x - x.min()) ** 2
#             fig = plt.figure(figsize=(14, 7))
#             for i in range(len(hyp)):
#                 y = a[:, i + 5]
#                 mu = (y * weights).sum() / weights.sum()
#                 plt.subplot(2, 5, i+1)
#                 plt.plot(x.max(), mu, 'o')
#                 plt.plot(x, y, '.')
#                 print(list(hyp.keys())[i],'%.4g' % mu)
