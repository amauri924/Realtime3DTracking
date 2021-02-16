import argparse
#import json

from torch.utils.data import DataLoader
from tqdm import tqdm
from models import *
from utils.datasets import *
from utils.utils import *
import math

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

def get_alpha(rot):
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    idx = rot[:, 1] > rot[:, 5]
    alpha1=np.zeros(len(rot[:, 2]))
    alpha2=np.zeros(len(rot[:, 2]))
    
    for i in range(len(alpha1)):
        alpha1_mod=np.arctan2(rot[i, 2] , rot[i, 3])- 1/2*np.pi
        alpha1[i] = alpha1_mod.item()
    
        alpha2_mod=np.arctan2(rot[i, 6] , rot[i, 7]) + 1/2*np.pi
        alpha2[i] = alpha2_mod.item()
    return alpha1 * idx + alpha2 * (1 - idx)


def get_orientation_score(alpha_pd,alpha_gt):
    OS=abs(np.cos(alpha_gt - alpha_pd))
    delta_theta=np.arccos(2*OS-1.0)*180/np.pi
    return OS,delta_theta

def roi_score(roi_pd,roi_gt,img_size):
    RS=(4+torch.cos((roi_gt[0]-roi_pd[0])/img_size)+torch.cos((roi_gt[1]-roi_pd[1])/img_size)+torch.cos((roi_gt[2]-roi_pd[2])/img_size)+torch.cos((roi_gt[3]-roi_pd[3])/img_size))/16
    return RS

def compute_classification_orient(t_alpha,orient_pred):
    loss_res=torch.cuda.FloatTensor([0],device=orient_pred.device)
    
    
    #Create 2 bin for alpha, bin covers 210° each
    #Bin centers are 0° and 180°
    bin_centers=torch.tensor([0,np.pi])
    
    res_gt=torch.zeros([t_alpha.shape[0],2],device=orient_pred.device)
    bin_gt=torch.zeros([t_alpha.shape[0],2],device=orient_pred.device).long()
    bin_pred=torch.zeros([t_alpha.shape[0],1],device=orient_pred.device).long()
    
    
    for idx in range(len(bin_gt)):
        alpha=t_alpha[idx]+1/2*np.pi
        
        
        #set alpha modulo pi
        alpha_mod=torch.tensor([alpha,alpha-2*np.pi,alpha+2*np.pi],device=orient_pred.device)
        alpha_mod=alpha_mod[torch.min(abs(alpha_mod), 0).indices]
        res_gt[idx,0]=alpha_mod
        if abs(alpha_mod)< 90*np.pi/180:
            bin_gt[idx,0]=1.0
        
        
        alpha=t_alpha[idx]-1/2*np.pi
        #set alpha modulo pi
        alpha_mod=torch.tensor([alpha,alpha+2*np.pi,alpha-2*np.pi],device=orient_pred.device)
        alpha_mod=alpha_mod[torch.min(abs(alpha_mod), 0).indices]
        res_gt[idx,1]=alpha_mod
        if abs(alpha_mod)< 90*np.pi/180:
            bin_gt[idx,1]=1.0
        
    idx= orient_pred[:, 1] > orient_pred[:, 5]
    for i in range(len(idx)):
        if int(idx[i].cpu().item()) == int(bin_gt[i,0]):
            bin_pred[i]=1.0
        else:
            bin_pred[i]=0.0
    return bin_pred


def compute_center_and_depth_errors(center_pred,depth_pred,pred_dim,orient_pred,roi_pred,center_abs_err,depth_abs_err,dim_abs_err,
                                    orient_abs_err,bbox_abs_err,targets_,img_shape,default_dims_tensor,dim_abs_err_dict,orient_abs_err_dict,depth_abs_err_dict,
                                    bin_pred_error_list,associated):
        # Statistics per image
        indxs_target=torch.tensor([idx[0] for idx in associated],device=targets_.device)
        indxs_pred=torch.tensor([idx[1] for idx in associated],device=depth_pred.device)
        
        targets=torch.index_select(targets_,0,indxs_target)
        
        center_pred=torch.index_select(center_pred,0,indxs_pred)
        depth_pred=torch.index_select(depth_pred,0,indxs_pred)
        pred_dim=torch.index_select(pred_dim,0,indxs_pred)
        orient_pred=torch.index_select(orient_pred,0,indxs_pred)
        roi_pred=torch.index_select(torch.cat(roi_pred),0,indxs_pred)
        
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
        
        tdim=labels[:, 8:11]*200
        talpha=labels[:, 12:13]*2*np.pi
        
        
        
        bin_pred_error=compute_classification_orient(talpha,orient_pred)
        
        bin_pred_error_list.append(bin_pred_error.cpu())
        
        p_alpha=get_alpha(orient_pred.cpu().data.numpy())
        
        for idx_pred in range(len(center_pred)):
            
            # err_new_roi=roi_score(roi_pred[idx_pred,:4], tbox[idx_pred],height)
            iou = bbox_iou(roi_pred[idx_pred,:4], tbox[idx_pred])
            bbox_abs_err.append(iou)
            
            target_center=tcent[idx_pred]
            predicted_center=center_pred[idx_pred]
            
            gt_depth=tdepth[idx_pred]
            predicted_depth=depth_pred[idx_pred]

            obj_cls=int(tcls[idx_pred])
            gt_dim=tdim[idx_pred]
            predicted_dim=default_dims_tensor[obj_cls,:] - pred_dim[idx_pred,obj_cls,:]*200
            
            
            predicted_center=predicted_center
            predicted_dim=predicted_dim

            
            w_bbox=tbox[idx_pred][2].cpu().item()-tbox[idx_pred][0].cpu().item()
            h_bbox=tbox[idx_pred][3].cpu().item()-tbox[idx_pred][1].cpu().item()
            centerbbox_x=tbox[idx_pred][0].cpu().item()+w_bbox/2
            centerbbox_y=tbox[idx_pred][1].cpu().item()+h_bbox/2
            
            predicted_center[0]=predicted_center[0]*w_bbox+centerbbox_x
            predicted_center[1]=predicted_center[1]*h_bbox+centerbbox_y
            
            tmp_dim_abs_err=[]
            for k in range(len(gt_dim)):
                tmp_dim_abs_err.append(abs(abs(predicted_dim[k]-gt_dim[k])/(gt_dim[k])))
            mean_abs=torch.mean(torch.tensor(tmp_dim_abs_err))
            dim_abs_err.append(mean_abs)
            try:
                dim_abs_err_dict[obj_cls].append(mean_abs.item())
            except:
                dim_abs_err_dict[obj_cls]=[mean_abs.item()]
            
            
            center_abs_err.append(torch.mean(torch.tensor([abs(abs(predicted_center[0]-target_center[0])/target_center[0]),abs(abs(predicted_center[1]-target_center[1])/target_center[1])])))
            
            depth_abs=abs(abs(predicted_depth-gt_depth)/(gt_depth+0.00001))
            depth_abs_err.append(depth_abs)
            try:
                depth_abs_err_dict[obj_cls].append(depth_abs.item())
            except:
                depth_abs_err_dict[obj_cls]=[depth_abs.item()]
            
            talpha_mod=torch.tensor([talpha[idx_pred,0:1]-2*np.pi,talpha[idx_pred,0:1]+2*np.pi,talpha[idx_pred,0:1]])
            talpha_mod=talpha_mod[torch.min(abs(talpha_mod),0).indices].item()
            
            mean_abs,_=get_orientation_score(p_alpha[idx_pred],talpha_mod)

            orient_abs_err.append(torch.tensor(mean_abs))
            
            try:
                orient_abs_err_dict[obj_cls].append(mean_abs)
            except:
                orient_abs_err_dict[obj_cls]=[mean_abs]
            
            
        return center_abs_err,depth_abs_err,dim_abs_err,orient_abs_err,bbox_abs_err,dim_abs_err_dict,orient_abs_err_dict,depth_abs_err_dict,bin_pred_error_list

def compute_mean_errors_and_print(stats,center_abs_err,depth_abs_err,dim_abs_err,
                                  orient_abs_err,bbox_abs_err,dim_abs_err_dict,
                                  orient_abs_err_dict,depth_abs_err_dict,bin_pred_error_list,nc,names,seen):
    
    bin_error=torch.tensor([])
    for i in range(len(bin_pred_error_list)):
        bin_error=torch.cat((bin_error,bin_pred_error_list[i].flatten()))
    bin_precision=torch.sum(bin_error).item()/len(bin_error)
    with open("precision_bin.txt","a") as f:
        f.write(str(bin_precision)+'\n')
    if len(center_abs_err)>0:
        center_abs_err=torch.mean(torch.tensor(center_abs_err)[torch.isfinite(torch.tensor(center_abs_err))]).cpu().item()
        depth_abs_err=torch.mean(torch.tensor(depth_abs_err)[torch.isfinite(torch.tensor(depth_abs_err))]).cpu().item()
        dim_abs_err=torch.mean(torch.tensor(dim_abs_err)[torch.isfinite(torch.tensor(dim_abs_err))]).cpu().item()
        
        orient_abs_err=torch.mean(torch.tensor(orient_abs_err)[torch.isfinite(torch.tensor(orient_abs_err))]).cpu().item()
        
        bbox_abs_err=torch.mean(torch.tensor(bbox_abs_err)[torch.isfinite(torch.tensor(bbox_abs_err))]).cpu().item()
        if math.isnan(depth_abs_err):
            print("nan")

    else:
        center_abs_err=0
        depth_abs_err=0
        dim_abs_err=0
    
    
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()

    # Print results
    pf = '%30s' + '%10.3g' * 10  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1, dim_abs_err, orient_abs_err, depth_abs_err,bbox_abs_err))
    with open("output.txt","a") as f:
        f.write(pf % ('all', seen, nt.sum(), mp, mr, map, mf1, dim_abs_err, orient_abs_err, depth_abs_err, bbox_abs_err)+'\n')

    # Print results per class
    pf = '%30s' + '%10.3g' * 9
    if nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i], np.mean(np.array(dim_abs_err_dict[c])), np.mean(np.array(orient_abs_err_dict[c])), np.mean(np.array(depth_abs_err_dict[c])) ))
            with open("output.txt","a") as f:
                f.write(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i], np.mean(np.array(dim_abs_err_dict[c])), np.mean(np.array(orient_abs_err_dict[c])), np.mean(np.array(depth_abs_err_dict[c])) )+'\n')


    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    
    
    # Return results
    return (mp, mr, map, mf1, center_abs_err,depth_abs_err,dim_abs_err,orient_abs_err), maps

def compute_bbox_error(output,targets,stats,width,height,iou_thres,seen):
    associated=[]
    len_target=0
    len_pred=0
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
                        

                        associated.append((bi.cpu().item()+len_target,i+len_pred))

            len_target+=len(labels)
            len_pred+=len(output[si])
            
            
            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))
    return stats,seen,associated

def test(
        cfg,
        data_cfg,
        weights=None,
        batch_size=16,
        img_size=416,
        iou_thres=0.3,
        conf_thres=0.001,
        nms_thres=0.3,
        save_json=False,
        model=None
):
    if model is None:
        device = torch_utils.select_device()

        # Initialize model
        model =  model = Model(cfg,0,transfer=False).to(device)
        

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
    print(('%30s' + '%10s' * 9) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1', 'dim_abs', 'alpha_abs','depth_abs'))
    with open("output.txt","a") as f:
        f.write(('%30s' + '%10s' * 9) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1', 'dim_abs', 'alpha_abs','depth_abs')+'\n')

    
    loss, p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    center_abs_err=[]
    depth_abs_err=[]
    dim_abs_err=[]
    dim_abs_err_dict={}
    orient_abs_err_dict={}
    depth_abs_err_dict={}
    orient_abs_err=[]
    bin_pred_error_list=[]
    bbox_abs_err=[]
    with open("data/KITTI/avg_shapes.json","r") as f:
        default_dims=json.load(f)
    default_dims_tensor=torch.zeros(len(default_dims),3,device=device)
    for class_idx in default_dims:
        default_dims_tensor[int(class_idx),:]=torch.tensor([shape for shape in default_dims[class_idx]])
    
    for batch_i, (imgs, targets, paths, shapes,_,_,_) in enumerate(tqdm(dataloader)):
        if len(targets)==0:
            print("skipping empty target")
            continue
        input_targets,width,height,imgs,targets=prepare_data_for_foward_pass(targets,device,imgs)
        with torch.cuda.amp.autocast():
            output_roi,center_pred, depth_pred ,pred_dim,orient_pred,new_rois= model(imgs,conf_thres=conf_thres, nms_thres=nms_thres,detect=True)  # inference and training outputs

        stats,seen,associated=compute_bbox_error(output_roi,targets,stats,width,height,0.5,seen)
        center_abs_err,depth_abs_err,dim_abs_err,orient_abs_err,bbox_abs_err,dim_abs_err_dict, orient_abs_err_dict, depth_abs_err_dict,bin_pred_error_list=compute_center_and_depth_errors(center_pred,depth_pred,pred_dim,
                                                                        orient_pred,new_rois,center_abs_err,depth_abs_err,
                                                                        dim_abs_err,orient_abs_err,bbox_abs_err,targets,(width,height),
                                                                        default_dims_tensor,dim_abs_err_dict, orient_abs_err_dict,
                                                                        depth_abs_err_dict,bin_pred_error_list,associated)

    (mp, mr, map, mf1, center_abs_err,
     depth_abs_err,dim_abs_err,orient_abs_err), maps=compute_mean_errors_and_print(stats,center_abs_err,depth_abs_err,
                                                                                   dim_abs_err,orient_abs_err,bbox_abs_err,dim_abs_err_dict,
                                                                                   orient_abs_err_dict,depth_abs_err_dict,bin_pred_error_list,nc,names,seen)
    return (mp, mr, map, mf1, center_abs_err,depth_abs_err,dim_abs_err,orient_abs_err), maps





if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=10, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-3dcent-KITTI.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/KITTI.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.3, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', default=False, help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
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
