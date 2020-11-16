import argparse
#import json

from torch.utils.data import DataLoader
from tqdm import tqdm
from models import *
from utils.datasets import *
from utils.utils import *

def prepare_data_for_foward_pass(targets,device,imgs):
    _, _, height, width = imgs.shape  # batch size, channels, height, width
    input_targets=targets.clone()
    targets = targets.to(device)
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
    imgs = imgs.to(device,non_blocking=True)
    input_targets = input_targets.to(device,non_blocking=True)
    return input_targets,width,height,imgs,targets


def compute_center_and_depth_errors(center_pred,depth_pred,pred_dim,center_abs_err,depth_abs_err,dim_abs_err,targets,img_shape):
        # Statistics per image
        labels=targets[:,1:]
        tcls=labels[:,0]
        width,height=img_shape
        tbox = xywh2xyxy(labels[:, 1:5])
        tbox[:, [0, 2]] *= width
        tbox[:, [1, 3]] *= height
        
        tcent=labels[:, 5:7]
        tdepth=labels[:,7]

        tcent[:, 0] *= width
        tcent[:, 1] *= height
        
        tdim=labels[:, 8:14]

        for idx_pred in range(len(center_pred)):
            target_center=tcent[idx_pred]
            predicted_center=center_pred[idx_pred]
            
            gt_depth=tdepth[idx_pred]
            predicted_depth=depth_pred[idx_pred]
            
            gt_dim=tdim[idx_pred]
            predicted_dim=pred_dim[idx_pred]
            
            obj_cls=int(tcls[idx_pred])
            predicted_center=predicted_center[obj_cls:obj_cls+2]
            predicted_dim=predicted_dim[obj_cls:obj_cls+6]

            
            w_bbox=tbox[idx_pred][2].cpu().item()-tbox[idx_pred][0].cpu().item()
            h_bbox=tbox[idx_pred][3].cpu().item()-tbox[idx_pred][1].cpu().item()
            centerbbox_x=tbox[idx_pred][0].cpu().item()+w_bbox/2
            centerbbox_y=tbox[idx_pred][1].cpu().item()+h_bbox/2
            
            predicted_center[0]=predicted_center[0]*w_bbox+centerbbox_x
            predicted_center[1]=predicted_center[1]*h_bbox+centerbbox_y
            
            tmp_dim_abs_err=[]
            for k in range(len(gt_dim)):
                tmp_dim_abs_err.append(abs(abs(predicted_dim[k]-gt_dim[k])/(gt_dim[k])))
            dim_abs_err.append(torch.mean(torch.tensor(tmp_dim_abs_err)))
                
            center_abs_err.append(torch.mean(torch.tensor([abs(abs(predicted_center[0]-target_center[0])/target_center[0]),abs(abs(predicted_center[1]-target_center[1])/target_center[1])])))
            depth_abs_err.append(abs(abs(predicted_depth-gt_depth)/(gt_depth+0.00001)))
        return center_abs_err,depth_abs_err,dim_abs_err

def compute_mean_errors_and_print(stats,center_abs_err,depth_abs_err,dim_abs_err,nc,names,seen):
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()

    # Print results
    pf = '%30s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    
    
    # Return results
    if len(center_abs_err)>0:
        center_abs_err=torch.mean(torch.tensor(center_abs_err)[torch.isfinite(torch.tensor(center_abs_err))]).cpu().item()
        depth_abs_err=torch.mean(torch.tensor(depth_abs_err)[torch.isfinite(torch.tensor(depth_abs_err))]).cpu().item()
        dim_abs_err=torch.mean(torch.tensor(dim_abs_err)[torch.isfinite(torch.tensor(dim_abs_err))]).cpu().item()
        if math.isnan(depth_abs_err):
            print("nan")

    else:
        center_abs_err=0
        depth_abs_err=0
        dim_abs_err=0
    
    return (mp, mr, map, mf1, center_abs_err,depth_abs_err,dim_abs_err), maps

def compute_bbox_error(output,targets,stats,width,height,iou_thres,seen):
    for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if len(pred)==0:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue


            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= width
                tbox[:, [1, 3]] *= height
                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor).nonzero().view(-1)
                    iou, bi = bbox_iou(pbox, tbox[m]).max(0)
                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and m[bi] not in [ind[1] for ind in detected]:  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append((i,m[bi].cpu().item()))
                
            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))
    return stats,seen

def test(
        cfg,
        data_cfg,
        weights=None,
        batch_size=16,
        img_size=416,
        iou_thres=0.6,
        conf_thres=0.001,
        nms_thres=0.6,
        save_json=False,
        model=None
):
    if model is None:
        device = torch_utils.select_device()

        # Initialize model
        model = Darknet(cfg, img_size).to(device)
        

        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        if torch.cuda.device_count() > 0:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device

    # Configure run
#    if type(model) is nn.parallel.DistributedDataParallel:
#        model.module.transfer=False
#    else:
#        model.transfer=False
    data_cfg = parse_data_cfg(data_cfg)
    nc = int(data_cfg['classes'])  # number of classes
    test_path = data_cfg['valid']  # path to test images
    names = load_classes(data_cfg['names'])  # class names
    available_cpu=len(os.sched_getaffinity(0))


    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size, batch_size,rect=False)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=available_cpu,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    seen = 0
    print("model device:"+str(device))
    model.eval()
    coco91class = coco80_to_coco91_class()
    print(('%30s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1'))
    loss, p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    center_abs_err=[]
    depth_abs_err=[]
    dim_abs_err=[]
    for batch_i, (imgs, targets, paths, shapes,_,_) in enumerate(tqdm(dataloader)):
        if len(targets)==0:
            print("skipping empty target")
            continue
        input_targets,width,height,imgs,targets=prepare_data_for_foward_pass(targets,device,imgs)
        
        output_roi,center_pred, depth_pred ,pred_dim= model(imgs,conf_thres=conf_thres, nms_thres=nms_thres,testing=True,targets=input_targets[:,np.array([0, 2, 3,4,5 ])])  # inference and training outputs

        center_abs_err,depth_abs_err,dim_abs_err=compute_center_and_depth_errors(center_pred,depth_pred,pred_dim,center_abs_err,depth_abs_err,dim_abs_err,targets,(width,height))
        stats,seen=compute_bbox_error(output_roi,targets,stats,width,height,iou_thres,seen)
    (mp, mr, map, mf1, center_abs_err,depth_abs_err,dim_abs_err), maps=compute_mean_errors_and_print(stats,center_abs_err,depth_abs_err,dim_abs_err,nc,names,seen)
    return (mp, mr, map, mf1, center_abs_err,depth_abs_err,dim_abs_err), maps





if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=12, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-3dcent-NS.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/3dcent-NS.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/best_1.1.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.6, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', default=False, help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        mAP = test(opt.cfg,
                   opt.data_cfg,
                   opt.weights,
                   opt.batch_size,
                   opt.img_size,
                   opt.iou_thres,
                   opt.conf_thres,
                   opt.nms_thres,
                   opt.save_json)
