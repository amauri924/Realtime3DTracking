import argparse
import time

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torch
import math
from tqdm import tqdm
import test as test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
import sys
#import torch_optimizer as optim
import torch.multiprocessing as mp
import json
from pathlib import Path
import yaml

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

#sys.path.append("/save/2020010/amauri03/package_perso")

#      0.109      0.297       0.15      0.126       7.04      1.666      4.062     0.1845       42.6       3.34      12.61      8.338     0.2705      0.001         -4        0.9     0.0005   320 giou + best_anchor False
hyp = {'giou': 1.666,  # giou loss gain
       'xy': 4.062,  # xy loss gain
       'wh': 0.1845,  # wh loss gain
       'cls': 42.6,  # cls loss gain
       'cls_pw': 3.34,  # cls BCELoss positive_weight
       'obj': 12.61,  # obj loss gain
       'obj_pw': 8.338,  # obj BCELoss positive_weight
       'orient':2, # orientation loss gain
       'depth':2, # depth loss gain
       'iou_t': 0.2705,  # iou target-anchor training threshold
      'max_lr': 5e-5,  # initial learning rate (SGD=1E-2, Adam=1E-3)
      'div_factor': 25,  # final OneCycleLR learning rate (lr0 * lrf)
        'final_div_factor': 1e4,  # final OneCycleLR learning rate (lr0 * lrf)
       'lr0': 1e-3,  # initial learning rate
       'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.90,  # SGD momentum
       'weight_decay': 0.0005}  # optimizer weight decay

real_hyp= {'orient':2, # orientation loss gain
            'depth':2}  # final OneCycleLR learning rate (lr0 * lrf)  # optimizer weight decay

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

def compute_rel_err(pred,target):
    depth_target=target[:,8:9].clone().detach()*200
    pred_depth=pred.clone().detach()*200
    
    rel_err=abs(depth_target-pred_depth)/depth_target
    
    return rel_err.split(1)

def check_updated_grad(previous_state_dict,model):
    list_k=[k for k,v in previous_state_dict.items()]
    previous_model_state={k: v for k,v in previous_state_dict.items()}
    actual_model_state={k: v for k,v in model.state_dict().items()}
    for k in actual_model_state:
        if not torch.all(torch.eq(actual_model_state[k], previous_model_state[k])).cpu().item():
            print("%s updated"%k)

def print_batch_results(mloss,i,loss_items,loss_scheduler,epoch,epochs,optimizer,nb,targets,img_size,log_path,rank):
    # Print batch results
    mloss = (mloss * i + loss_items.cpu().detach()) / (i + 1)  # update mean losses
    loss_scheduler.append(mloss[-1])
    for x in optimizer.param_groups:
        s = ('%8s%12s' + '%10.3g' * 13) % (
            '%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, nb - 1), *mloss, len(targets), img_size, x['lr'])
    with open(log_path, 'a') as logfile:
        logfile.write(str(rank)+" "+s+"\n")
    return mloss,loss_scheduler,s



def prepare_data_before_forward(imgs,device,targets,augmented_roi,i,nb,epoch,accumulate,multi_scale,img_size_min,img_size_max,idx_train,paths,img_size,pixel_to_normalized_resized):
    input_targets=augmented_roi.clone()
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
    resize_matrix[:,0,:]*=imgs.shape[2]
    resize_matrix[:,1,:]*=imgs.shape[3]
    
    imgs = imgs.to(device=device,non_blocking=True)
    imgs=imgs.contiguous(memory_format=torch.channels_last)
    input_targets = input_targets.to(device,non_blocking=True)
    
    
    return imgs,targets,input_targets,img_size,resize_matrix

def run_test_and_save(model,opt,optimizer,s,data_cfg,batch_size,cfg,log_path,result_path,best_fitness,epoch,epochs,latest,best):
    with open(log_path, 'a') as logfile:
        logfile.write("testing \n")
    with torch.no_grad():
        
        if type(model) is nn.parallel.DistributedDataParallel:
            results,_ = test.test(cfg, data_cfg, batch_size=batch_size, img_size=opt.img_size,
                                      model=model.module,
                                      conf_thres=0.1)
#            with open("debug.txt","a") as f:
#                f.write("Testing on the train split\n")
#            results_training,_ = test.test(cfg, 'data/3dcent-NS-training.data', batch_size=batch_size, img_size=opt.img_size,
#                                      model=model.module,
#                                      conf_thres=0.1)
        else:
            results,_ = test.test(cfg, data_cfg, batch_size=batch_size, img_size=opt.img_size,
                                      model=model,
                                      conf_thres=0.1)
#            results_training,_ = test.test(cfg, 'data/3dcent-NS-training.data', batch_size=batch_size, img_size=opt.img_size,
#                                      model=model,
#                                      conf_thres=0.1)
    # Write epoch results
    with open(result_path, 'a') as file:
        file.write(s +'|||'+ '%11.3g' * 8 % results + '\n')  # P, R, mAP, F1, center_abs_err, depth_abs_err, dim_abs_err, orient_abs_err
    

    
    # Update best map
    fitness = results[5]+results[7]
    if fitness < best_fitness and fitness!=0:
        print("best error replaced by %f"%fitness)
        best_fitness = fitness
    save = not opt.nosave
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

#
        # Save best checkpoint
        if not math.isnan(fitness):
            if best_fitness == fitness:
                torch.save(chkpt, best)
        
    return results,best_fitness


def train(cfg,
          real_hyp,
        data_cfg,
        opt,
        img_size=416,
        epochs=100,  # 500200 batches at bs 16, 117263 images = 273 epochs
        batch_size=16,
        accumulate=4,  # effective bs = batch_size * accumulate = 8 * 8 = 64
        freeze_backbone=False,
        world_size=1,
        rank=0
):
    idx_train=str(batch_size)+'.'+str(accumulate)
    log_path='log_'+idx_train+str(opt.run_id)+'.txt'
    result_path='result_'+idx_train+str(opt.run_id)+'.txt'
    init_seeds()
    weights = 'weights' + os.sep
    latest = weights + 'latest.pt'
    best = weights + 'best.pt'
    device = rank
    print(device)
    device_loading= torch.tensor([]).to(device).device
    multi_scale = True
    available_cpu=12

    print('num_cores_available:',available_cpu)

    if multi_scale:
        img_size_min = round(img_size / 32 / 2)
        img_size_max = round(img_size / 32)
        img_size = img_size_max * 32  # initiate with maximum multi_scale size
    else:
        img_size_min=round(img_size / 32)
        img_size_max=round(img_size / 32)
        
        
    # Configure run
    data_dict = parse_data_cfg(data_cfg)
    train_path = data_dict['train']
    nc = int(data_dict['classes'])  # number of classes

    # Initialize model
    model = Model(cfg,hyp,real_hyp,transfer=False).to(device,memory_format=torch.channels_last)

    # Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),
                           lr=1e-4,  weight_decay=3e-6)
    scaler = torch.cuda.amp.GradScaler()
    
    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_fitness = 1e6
    if opt.resume or opt.transfer:  # Load previously saved model
        if opt.transfer and not opt.resume:  # Transfer learning
#            nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)
            chkpt = torch.load(weights + 'yolov3.pt', map_location=device_loading)
            model_state=[k for k,v in model.Yolov3.state_dict().items()]
            chkp_state=[k for k,v in chkpt['model'].items()]

            new_state={}
            for k,v in chkpt['model'].items():
                layer=k.split('.')[1]
                if k.split('.')[2]=="Conv2d":
                    new_k=k.split('.')[0]+'.'+k.split('.')[1]+'.'+'conv_'+layer+'.'+k.split('.')[3]
                if k.split('.')[2]=="BatchNorm2d":
                    new_k=k.split('.')[0]+'.'+k.split('.')[1]+'.'+'batch_norm_'+layer+'.'+k.split('.')[3]
                new_state[new_k]= v
            tmp={k: new_state[k] for k in new_state if k in model_state}
            new_state={k: tmp[k] for k in tmp if model.Yolov3.state_dict()[k].shape==tmp[k].shape}
            model.Yolov3.load_state_dict({k: new_state[k] for k in new_state},strict=False)
            del new_state

        else:  # resume from latest.pt
            chkpt = torch.load(latest, map_location=device_loading)  # load checkpoint
            model.load_state_dict(chkpt['model'])

        if chkpt['optimizer'] is not None:
            if opt.resume:
                optimizer.load_state_dict(chkpt['optimizer'])
                try:
                    best_fitness = chkpt['best_fitness']
                except:
                    best_fitness = 1e6
                    
        try:
            if chkpt['training_results'] is not None:
                with open('results_'+str(opt.run_id)+'.txt', 'w') as file:
                    file.write(chkpt['training_results'])  # write results.txt
        except:
            print("no previous training results")

        start_epoch = chkpt['epoch'] + 1
        del chkpt


        # Remove old results
        for f in glob.glob('*_batch*.jpg') + glob.glob('results.txt'):
            os.remove(f)





    # Dataset
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  augment=True,
                                  rect=opt.rect,
                                  depth_aug=opt.depth_aug)  # rectangular training

    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                num_replicas=world_size,
                                                                rank=rank
                                                                )
    
    
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=available_cpu,
                            shuffle=False,  # Shuffle=True unless rectangular training is used
                            pin_memory=True,
                            collate_fn=dataset.collate_fn,
                            sampler=train_sampler)


    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 5e-5, epochs=epochs, steps_per_epoch=int(len(dataloader)/accumulate)+1)
    scheduler.last_epoch = start_epoch - 1



    # Initialize distributed training
#    if torch.cuda.device_count() > 1:
#        with open(log_path, 'w') as logfile:
#            logfile.write("nb GPU : %i\n"%torch.cuda.device_count())
    # dist.init_process_group(backend='nccl',  # 'distributed backend'
    #                         world_size=world_size,  # number of nodes for distributed training
    #                         rank=rank)  # distributed training node rank

    # model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[rank],find_unused_parameters = True)





    # Start training
    model_info(model, report='summary')  # 'full' or 'summary'
    nb = len(dataloader)
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    t0 = time.time()
    with open(log_path, 'a') as logfile:
        logfile.write(str(rank)+" "+"nb epochs : %i\n"%epochs)
        
    with open("data/3dcent-NS/avg_shapes.json","r") as f:
        default_dims=json.load(f)
    default_dims_tensor=torch.zeros(len(default_dims),3,device=device)
    for class_idx in default_dims:
        default_dims_tensor[int(class_idx),:]=torch.tensor([shape for shape in default_dims[class_idx]])
    
    
    for epoch in range(start_epoch, epochs):
        if epoch < -1:
            opt.notest = True
        else:
            opt.notest = False
        
        loss_scheduler=[]
        model.train()
        
        model.Yolov3.eval()
        
        rel_err=[]
        #set new learning rate
#        for param_group in optimizer.param_groups:
#            param_group['lr'] = lr_values[epoch]

        with open(log_path, 'a') as logfile:
            logfile.write(str(rank)+" "+"Epoch: %i\n"%epoch)


        mloss = torch.zeros(10)  # mean losses

        for i, (imgs, targets, paths, _,calib,pixel_to_normalized_resized,augmented_roi) in enumerate(tqdm(dataloader)):
            
            if len(targets)>0:
                # if opt.depth_aug:
                #     targets=rois_augmentation_for_depth(targets,0.2,0.02)
                #Prepare data
                imgs,targets,input_targets,img_size,resize_matrix=prepare_data_before_forward(imgs,device,targets,augmented_roi,i,nb,epoch,accumulate,
                                            multi_scale,img_size_min,img_size_max,idx_train,paths,img_size,pixel_to_normalized_resized)
                
                # Run model
                with torch.cuda.amp.autocast():
                    pred,pred_center,depth_pred,dim_pred,orient_pred,orient_bin_cls = model(imgs,targets=input_targets[:,np.array([0, 2, 3,4,5 ])])
                    loss, loss_items = compute_loss(pred,pred_center,depth_pred,dim_pred,orient_pred,orient_bin_cls, targets, model,imgs.shape[2:],calib,resize_matrix,default_dims_tensor, giou_loss=not opt.xywh,rank=device)
               
                # Compute gradient
                scaler.scale(loss).backward()
              
                #Depth relative error
                rel_err_=compute_rel_err(depth_pred.float(),targets.float())
                for value in rel_err_:
                    rel_err.append(value.float())
               
                
                if torch.isnan(loss):
                    with open(log_path, 'a') as logfile:
                        logfile.write('WARNING: nan loss detected, ending training \n')
                    return pred
                
                mloss,loss_scheduler,s=print_batch_results(mloss,i,loss_items,loss_scheduler,epoch,epochs,optimizer,nb,targets,img_size,log_path,rank)
                
            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == nb:
                scaler.step(optimizer)
                scaler.update() 
                for param in model.parameters():
                    param.grad = None
                scheduler.step()

        if rank == 0:
            for param in model.parameters():
                param.grad = None
            rel_err=torch.mean(torch.tensor(rel_err))
            with open("depth_rel_err.txt","a") as f:
                f.write("rel err:"+str(rel_err)+'\n')
            loss_scheduler=torch.mean(torch.tensor(loss_scheduler)).item()
            with open("epoch_loss.txt","a") as f:
                f.write("Epoch loss:"+str(loss_scheduler)+'\n')
            new_lr=scheduler.get_last_lr()
            with open("lr.txt","a") as f:
#                f.write("LR:"+str(optimizer.param_groups[0]['lr'])+'\n')
                f.write("LR:"+str(new_lr)+'\n')
                
            # Report time
            dt = (time.time() - t0) / 3600
            with open(log_path, 'a') as logfile:
                logfile.write('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, dt))
    
            chkpt = {'epoch': epoch,
                 'model': model.module.state_dict() if type(
                     model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                 'optimizer': optimizer.state_dict()}

            # Save latest checkpoint
            torch.save(chkpt, latest)
            # Calculate mAP (always test final epoch, skip first 5 if opt.nosave)
            if not opt.notest:
                results,best_fitness=run_test_and_save(model,opt,optimizer,s,data_cfg,batch_size,cfg,log_path,result_path,best_fitness,epoch,epochs,latest,best)

    with open(log_path, 'a') as logfile:
        logfile.write("ending \n")
        
        
    torch.cuda.empty_cache()
    return results


def main(opt):
    world_size=torch.cuda.device_count()
    # mp.spawn(example,
    #     args=(world_size,opt),
    #     nprocs=world_size,
    #     join=True)
    example(0,world_size,opt)
    
def example(rank, world_size,opt):
    results=train(opt.cfg,
                  hyp,
                opt.data_cfg,
                opt,
                img_size=opt.img_size,
                epochs=opt.epochs,
                batch_size=opt.batch_size,
                accumulate=opt.accumulate,
                world_size=world_size,
                rank=rank
                )

def fitness(x):
    w = [0.0, 0.0, 0.0, 0.0,0.1,0.4,0.0,0.6] # weights for [mp, mr, map, mf1, center_abs_err,depth_abs_err,dim_abs_err,orient_abs_err]
    return ((1-x[:, :8]) * w).sum(1) 

def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=''):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        url = 'gs://%s/evolve.txt' % bucket
        if gsutil_getsize(url) > (os.path.getsize('evolve.txt') if os.path.exists('evolve.txt') else 0):
            os.system('gsutil cp %s .' % url)  # download evolve.txt if larger than local

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    x = x[np.argsort(-fitness(x))]  # sort
    np.savetxt('evolve.txt', x, '%10.3g')  # save sort by fitness

    # Save yaml
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])
    with open(yaml_file, 'w') as f:
        results = tuple(x[0, :7])
        c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)

    if bucket:
        os.system('gsutil cp evolve.txt %s gs://%s' % (yaml_file, bucket))  # upload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', default='', help='number of epochs')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--accumulate', type=int, default=4, help='number of batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/3dcent-COCO.data', help='coco.data file path')
    parser.add_argument('--multi-scale', default=True, help='train at (1/1.5)x - 1.5x sizes')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--rect', default=False, help='rectangular training')
    parser.add_argument('--resume', default=False, help='resume training flag')
    parser.add_argument('--depth_aug', default=True, help='resume training flag')
    parser.add_argument('--transfer', default=True, help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=7, help='number of Pytorch DataLoader workers')
    parser.add_argument('--nosave', default=False, help='only save final checkpoint')
    parser.add_argument('--notest', default=False, help='only test final epoch')
    parser.add_argument('--xywh', action='store_true', help='use xywh loss instead of GIoU loss')
    parser.add_argument('--evolve', default=True, help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--var', default=0, type=int, help='debug variable')
    opt = parser.parse_args()
    print(opt)

    if opt.evolve:
        opt.notest = True  # only test final epoch
        opt.nosave = True  # only save final checkpoint
        
        meta = {'orient':(1, 0.01, 1.0), # orientation loss gain
                'depth':(1, 0.01, 1.0)} # depth loss gain}  # image mixup (probability)
        
        yaml_file = Path('hyp_evolved.yaml')  # save best result here
        
        for _ in range(300):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(real_hyp.keys()):  # plt.hist(v.ravel(), 300)
                    real_hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                real_hyp[k] = max(real_hyp[k], v[1])  # lower limit
                real_hyp[k] = min(real_hyp[k], v[2])  # upper limit
                real_hyp[k] = round(real_hyp[k], 5)  # significant digits

            # Train mutation
            results = train(opt.cfg,
                            real_hyp,
                            opt.data_cfg,
                            opt,
                            img_size=opt.img_size,
                            epochs=opt.epochs,
                            batch_size=opt.batch_size,
                            accumulate=opt.accumulate,
                            world_size=torch.cuda.device_count(),
                            rank=0)

            # Write mutation results
            print_mutation(real_hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print('Hyperparameter evolution complete. Best results saved as: %s\nCommand to train a new model with these '
              'hyperparameters: $ python train.py --hyp %s' % (yaml_file, yaml_file))
        
        
        
    else:
        main(opt)


    # Train
#    results = train(opt.cfg,
#                    opt.data_cfg,
#                    img_size=opt.img_size,
#                    epochs=opt.epochs,
#                    batch_size=opt.batch_size,
#                    accumulate=opt.accumulate
#                    )

