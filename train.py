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
       'lr0': 0.001,  # initial learning rate
       'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.90,  # SGD momentum
       'weight_decay': 0.0005}  # optimizer weight decay


#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def check_updated_grad(previous_state_dict,model):
    list_k=[k for k,v in previous_state_dict.items()]
    previous_model_state={k: v for k,v in previous_state_dict.items()}
    actual_model_state={k: v for k,v in model.state_dict().items()}
    for k in actual_model_state:
        if not torch.all(torch.eq(actual_model_state[k], previous_model_state[k])).cpu().item():
            print("%s updated"%k)

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
    best_fitness = 1000
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


#    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in (0.8, 0.9)], gamma=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True)
    scheduler.last_epoch = start_epoch - 1



    # Dataset
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  augment=True,
                                  rect=opt.rect)  # rectangular training

    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        with open(log_path, 'w') as logfile:
            logfile.write("nb GPU : %i\n"%torch.cuda.device_count())
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
    
        model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters = False)
#        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=opt.num_workers,
                            shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)
#                            sampler=sampler)

    # Mixed precision training https://github.com/NVIDIA/apex
    mixed_precision = True
    if mixed_precision:
        try:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
            print("using mixed precision")
        except:  # not installed: install help: https://github.com/NVIDIA/apex/issues/259
            mixed_precision = False

    # Start training
#    model.hyp = hyp  # attach hyperparameters to model
    # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model_info(model, report='summary')  # 'full' or 'summary'
    nb = len(dataloader)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
#    n_burnin = min(round(nb / 5 + 1), 1000)  # burn-in batches
    n_burnin=-1
    t, t0 = time.time(), time.time()
    with open(log_path, 'a') as logfile:
        logfile.write("nb epochs : %i\n"%epochs)
        
        #Freezing layer


    for epoch in range(start_epoch, epochs):
        loss_scheduler=[]
        model.train()
        if opt.transfer:
            model.transfer=True
            model.eval()
            if type(model) is nn.parallel.DistributedDataParallel:
                model.module.depth_pred.train()
                model.module.featurePooling.train()
            else:
                model.depth_pred.train()
                model.featurePooling.train()
#                model.Yolov3.train()

        with open(log_path, 'a') as logfile:
            logfile.write("Epoch: %i"%epoch)

        # Update scheduler
#        if epoch>0:
#            scheduler.step(loss_scheduler)
        
        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        mloss = torch.zeros(8)  # mean losses
        
        
        
        for i, (imgs, targets, paths, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            
            if opt.depth_aug:
                targets=rois_augmentation_for_depth(targets,0.2,0.02)
                
            if imgs.shape[0]<torch.cuda.device_count():
                continue
            if imgs.shape[0]!=opt.batch_size:
                continue
#                imgs=imgs[:int(imgs.shape[0]/torch.cuda.device_count())*torch.cuda.device_count(),...]
#                targets=targets[:int(targets.shape[0]/torch.cuda.device_count())*torch.cuda.device_count(),...]

            if len(targets)==0:
                continue
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

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                lr = hyp['lr0'] * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr
            
            with open("log_time"+idx_train+str(opt.run_id)+".txt",'a') as f:
                f.write("step %i\n"%i)
                f.write('Paths:\n'+str(paths)+'\n')
            
            # Run model
            t_pred=time.time()
            pred,pred_center,depth_pred = model(imgs,targets=input_targets)
            t_pred=time.time()-t_pred
            
            with open("log_time"+idx_train+str(opt.run_id)+".txt",'a') as f:
                f.write("tpred %f\n"%t_pred)


            # Compute loss
            try:
                loss, loss_items = compute_loss(pred,pred_center,depth_pred, targets, model,imgs.shape[2:], giou_loss=not opt.xywh)
            except:
                with open("debug_"+str(opt.run_id)+".txt",'a') as f:
                    f.write("error in loss\n")
                continue
#            if loss.cpu().item()>100:
#                print("ehhh")
                
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
            
            # Print batch results
            mloss = (mloss * i + loss_items.cpu()) / (i + 1)  # update mean losses
            loss_scheduler.append(mloss[-1])
            # s = ('%8s%12s' + '%10.3g' * 7) % ('%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, nb - 1), *mloss, len(targets), time.time() - t)
            for x in optimizer.param_groups:
                s = ('%8s%12s' + '%10.3g' * 11) % (
                    '%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, nb - 1), *mloss, len(targets), img_size, x['lr'])
            del pred,pred_center
            t = time.time()
            with open(log_path, 'a') as logfile:
                    logfile.write(s+"\n")
#            pbar.set_description(s)  # print(s)

        loss_scheduler=torch.mean(torch.tensor(loss_scheduler)).item()
        print("Epoch loss:"+str(loss_scheduler))
        scheduler.step(loss_scheduler)
        # Report time
        dt = (time.time() - t0) / 3600
        with open(log_path, 'a') as logfile:
            logfile.write('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, dt))
#        torch.cuda.synchronize()

        # Calculate mAP (always test final epoch, skip first 5 if opt.nosave)
        if (not (opt.notest or (opt.nosave and epoch < 10)) and epoch%1==0) or (epoch == epochs - 1):
            with open(log_path, 'a') as logfile:
                logfile.write("testing \n")
            with torch.no_grad():
                
                if type(model) is nn.parallel.DistributedDataParallel:
                
                    results, maps = test.test(cfg, data_cfg, batch_size=3, img_size=opt.img_size,
                                              model=model,
                                              conf_thres=0.1)
                    results_training, _ = test.test(cfg, 'data/3dcent-NS-training.data', batch_size=3, img_size=opt.img_size,
                                              model=model,
                                              conf_thres=0.1)
                else:
                    results, maps = test.test(cfg, data_cfg, batch_size=1, img_size=opt.img_size,
                                              model=model,
                                              conf_thres=0.1)
                    results_training, _ = test.test(cfg, 'data/3dcent-NS-training.data', batch_size=1, img_size=opt.img_size,
                                              model=model,
                                              conf_thres=0.1)
            # Write epoch results
            with open(result_path, 'a') as file:
                file.write(s + '%11.3g' * 9 % results + '\n')  # P, R, mAP, F1, test_loss, center_abs_err, dconf_loss, depth_abs_err, "real" depth_abs_err

            with open("result_training"+str(opt.run_id)+".txt", 'a') as file:
                file.write('%g/%g' % (epoch, epochs - 1) + '%11.3g' * 4 % (results_training[-1], results[-1],mloss[-1].item(),results[2])+'\n')  #training_abs,test_abs,loss,mAP

            # Update best map
            fitness = results[-1]
            if fitness < best_fitness:
                print("best error replaced by %f"%fitness)
                best_fitness = fitness

        # Update best loss
#        fitness = results[4]
#        if not math.isnan(fitness):
#            if fitness < best_fitness:
#                best_fitness = fitness


        # Save training results
        save = ((not opt.nosave) and (epoch%10==0)) or ((not opt.evolve) and (epoch == epochs - 1))
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

            # Save backup every 50 epochs (optional)
#            if epoch > 0 and epoch % 50 == 0:
#                torch.save(chkpt, weights + 'backup%g.pt' % epoch)

        
        
#            torch.cuda.empty_cache()
        # Delete checkpoint
#        del chkpt
    with open(log_path, 'a') as logfile:
        logfile.write("ending \n")
    return results


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', default=str(time.time()).split(".")[0], help='number of epochs')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--accumulate', type=int, default=1, help='number of batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-3dcent-NS.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/3dcent-NS.data', help='coco.data file path')
    parser.add_argument('--multi-scale', default=True, help='train at (1/1.5)x - 1.5x sizes')
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--rect', default=False, help='rectangular training')
    parser.add_argument('--resume', default=False, help='resume training flag')
    parser.add_argument('--depth_aug', default=False, help='resume training flag')
    parser.add_argument('--transfer', default=True, help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=28, help='number of Pytorch DataLoader workers')
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
