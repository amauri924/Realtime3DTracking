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

def compute_loss(t_alpha,orient_pred):
    ft = torch.cuda.FloatTensor if t_alpha.is_cuda else torch.Tensor
    l_orientation=ft([0])
    rot_loss=compute_rot_loss(t_alpha,orient_pred,l_orientation)*33.3
    return rot_loss

def get_alpha(rot):
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    idx = rot[:, 1] > rot[:, 5]
    alpha1=np.zeros(len(rot[:, 2]))
    alpha2=np.zeros(len(rot[:, 2]))
    
    for i in range(len(alpha1)):
        alpha1_mod=np.arctan2(rot[i, 2] , rot[i, 3])- 0
        alpha1[i] = alpha1_mod.item()
    
        alpha2_mod=np.arctan2(rot[i, 6] , rot[i, 7]) - np.pi
        alpha2[i] = alpha2_mod.item()
    return alpha1 * idx + alpha2 * (1 - idx)

def display_3dbbox(img,theta,dim,center_3d,calib):
    
    width,height,length=dim
    K_model_to_cam=np.array([[-np.sin(theta),0,np.cos(theta),center_3d[0][0]],
                            [0,1,0,center_3d[1][0]],
                            [-np.cos(theta),0,-np.sin(theta),center_3d[2][0]],
                            [0,0,0,1]])
                
    x_min=-width/2
    x_max=width/2
    y_min=-height/2
    y_max=height/2
    z_min=-length/2
    z_max=length/2
    
    
    points_3dbbox_homo_object_coord = np.array([
    [x_min, y_min, z_min,1],
    [x_min, y_min, z_max,1],
    [x_min, y_max, z_min,1],
    [x_min, y_max, z_max,1],
    [x_max, y_min, z_min,1],
    [x_max, y_min, z_max,1],
    [x_max, y_max, z_min,1],
    [x_max, y_max, z_max,1],
    ]).T
    
    points_3dbbox_homo_cam_coord=K_model_to_cam@points_3dbbox_homo_object_coord
    pixel_points = calib.numpy()@points_3dbbox_homo_cam_coord
    pixel_points = pixel_points[:2,:] / pixel_points[2,:]
    
    pixel_points[1,:]=np.clip(pixel_points[1,:], 0, img.shape[0])
    pixel_points[0,:]=np.clip(pixel_points[0,:], 0, img.shape[1])
    
    bbox_2d = np.zeros((8, 2))
    for k in range(points_3dbbox_homo_cam_coord.shape[1]):
        
        bbox_2d[k, :] = pixel_points[:,k]
    # fig.savefig(save_path,dpi=200)
    bbox_2d=bbox_2d.round().astype(np.int32)
    # bbox_2d=bbox_2d.reshape(-1,1,2)
    
    cv2.polylines(img,[bbox_2d[(0, 1, 3, 2), :].reshape(-1,1,2)],False,(0,255,0))
    cv2.polylines(img,[bbox_2d[(4, 5, 7, 6), :].reshape(-1,1,2)],False,(0,255,0))
    cv2.polylines(img,[bbox_2d[(0, 2, 6, 4), :].reshape(-1,1,2)],False,(0,255,0))
    cv2.polylines(img,[bbox_2d[(1, 3, 7, 5), :].reshape(-1,1,2)],False,(0,255,0))
    cv2.polylines(img,[bbox_2d[(0, 1, 5, 4), :].reshape(-1,1,2)],False,(0,255,0))
    cv2.polylines(img,[bbox_2d[(2, 3, 7, 6), :].reshape(-1,1,2)],False,(0,255,0))

def get_theta_pred(roi_pred,selected_calib,center_pred,depth_pred,selected_shape,pred_alpha,default_dims_tensor,pred_dim,selected_path):
    selected_calib=selected_calib.cpu()
    theta_list=[]
    previous_path=""
    img=False
    for j,(*xyxy, conf, cls_conf, cls) in enumerate(roi_pred):
        
        xyxy=torch.tensor(xyxy).view(1,-1)
        xyxy = scale_coords([opt.img_size,opt.img_size], xyxy.cuda(), selected_shape[j]).round()
        xyxy=xyxy.view(-1)
        w_bbox=xyxy[2]-xyxy[0]
        h_bbox=xyxy[3]-xyxy[1]
        centerbbox_x=xyxy[0]+w_bbox/2
        centerbbox_y=xyxy[1]+h_bbox/2
        
        x=(center_pred[j][0]*w_bbox+centerbbox_x).item()
        y=(center_pred[j][1].item()*h_bbox+centerbbox_y).item()
        depth=depth_pred[j].item()*200
                    
        homo_center=np.array([[x*depth],[y*depth],[depth],[1]])
        center_3d=np.linalg.inv(selected_calib[j]) @ homo_center
        theta=pred_alpha[j]+math.atan2(center_3d[0][0],center_3d[2][0])
        theta_list.append(pred_alpha[j]+math.atan2(center_3d[0][0],center_3d[2][0]))
        
        if previous_path!=selected_path[j]:
            if j>0:
                cv2.imwrite("/home/antoine/Realtime3DTracking/output/"+previous_path.split("/")[-1],img)
            img=cv2.imread(selected_path[j])
        dim=(default_dims_tensor[int(cls),:] - pred_dim[j,int(cls),:]*200).cpu().numpy()
        display_3dbbox(img,theta,dim,center_3d,selected_calib[j])
        previous_path=selected_path[j]
        if j+1==len(roi_pred):
            cv2.imwrite("/home/antoine/Realtime3DTracking/output/"+selected_path[j].split("/")[-1],img)
    return theta_list

def get_orientation_score(alpha_pd,alpha_gt):
    # OS=abs(np.cos(alpha_gt - alpha_pd))
    OS=(1.0 + np.cos(alpha_gt - alpha_pd)) / 2.0
    delta_theta=np.arccos(2*OS-1.0)*180/np.pi
    return OS,delta_theta

def compute_depth_errors(gt, pred):
    assert (np.all(np.isfinite(gt) & np.isfinite(pred) & (gt > 0) & (pred > 0)))
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    log_diff = (np.log(gt) - np.log(pred))
    rmse_log = np.sqrt(np.square(log_diff).mean())

    scale_invariant = np.sqrt(
        np.square(log_diff).mean() - np.square(log_diff.mean()))

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, scale_invariant, a1, a2, a3



def compute_classification_orient(t_alpha,orient_pred):
    loss_res=torch.cuda.FloatTensor([0],device=orient_pred.device)
    
    
    #Create 2 bin for alpha, bin covers 210° each
    #Bin centers are 0° and 180°
    bin_centers=torch.tensor([0,np.pi])
    
    res_gt=torch.zeros([t_alpha.shape[0],2],device=orient_pred.device)
    bin_gt=torch.zeros([t_alpha.shape[0],2],device=orient_pred.device).long()
    bin_pred=torch.zeros([t_alpha.shape[0],1],device=orient_pred.device).long()
    
    
    for idx in range(len(bin_gt)):
        alpha=t_alpha[idx]+0
        
        
        #set alpha modulo pi
        alpha_mod=torch.tensor([alpha,alpha-2*np.pi,alpha+2*np.pi],device=orient_pred.device)
        alpha_mod=alpha_mod[torch.min(abs(alpha_mod), 0).indices]
        res_gt[idx,0]=alpha_mod
        if abs(alpha_mod)< 90*np.pi/180:
            bin_gt[idx,0]=1.0
        
        
        alpha=t_alpha[idx]-np.pi
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

def store_all_depths(depth_pred,targets,store_dict,associated):
    indxs_target=torch.tensor([idx[0] for idx in associated],device=targets.device)
    indxs_pred=torch.tensor([idx[1] for idx in associated],device=depth_pred.device)
        
    targets_selected=torch.index_select(targets,0,indxs_target)
    depth_pred=torch.index_select(depth_pred,0,indxs_pred)
    tcls=targets_selected[:,1]
    tdepth=targets_selected[:,8]
    
    num_cls=int(torch.max(tcls).cpu().item())
    
    for i in range(num_cls+1):
        idx_cls=(tcls==i).nonzero()[:,0]
        class_tdepth=torch.index_select(tdepth,0,idx_cls.T)
        class_pdepth=torch.index_select(depth_pred,0,idx_cls.T)
        try:
            store_dict[i][0]+= list(class_pdepth.cpu().numpy().T[0,:])
            store_dict[i][1]+= list(class_tdepth.cpu().numpy())
        except:
            store_dict[i]=[list(class_pdepth.cpu().numpy().T[0,:]),list(class_tdepth.cpu().numpy())]

    return store_dict

def compute_center_and_depth_errors(center_pred,depth_pred,pred_dim,orient_pred,center_abs_err,depth_abs_err,dim_abs_err,
                                    orient_abs_err,targets_,img_shape,default_dims_tensor,dim_abs_err_dict,orient_abs_err_dict,depth_abs_err_dict,
                                    bin_pred_error_list,associated,loss_list,det,calib,shapes,paths):
        # Statistics per image
        indxs_target=torch.tensor([idx[0] for idx in associated],device=targets_.device)
        indxs_pred=torch.tensor([idx[1] for idx in associated],device=depth_pred.device)
        
        targets=torch.index_select(targets_,0,indxs_target)
        
        target_batch_idx=targets[:,0].long()
        
        center_pred=torch.index_select(center_pred,0,indxs_pred)
        depth_pred=torch.index_select(depth_pred,0,indxs_pred)
        pred_dim=torch.index_select(pred_dim,0,indxs_pred)
        orient_pred=torch.index_select(orient_pred,0,indxs_pred)
        roi_pred=torch.index_select(torch.cat(det),0,indxs_pred)
        selected_calib=torch.index_select(calib,0,target_batch_idx)
        selected_shape=torch.index_select(torch.tensor(shapes).cuda(),0,target_batch_idx)
        selected_path=np.array(paths)[target_batch_idx.cpu().numpy()]
        
        
        labels=targets[:,1:]
        tcls=labels[:,0]
        width,height=img_shape
        tbox = xywh2xyxy(labels[:, 1:5])
        tbox[:, [0, 2]] *= width
        tbox[:, [1, 3]] *= height
        
        tcent=labels[:, 5:7]
        tdepth=labels[:,7]
        
        # abs_rel, sq_rel, rmse, rmse_log, scale_invariant, a1, a2, a3=compute_depth_errors(tdepth.cpu().numpy()*200, depth_pred[:,0].cpu().numpy().T*200)
        
        tcent[:, 0] *= width
        tcent[:, 1] *= height
        
        tdim=labels[:, 8:11]*200
        talpha=labels[:, 12:13]*2*np.pi
        
        loss=compute_loss(talpha/(2*np.pi),orient_pred)
        loss_list.append(loss.cpu().item())
        bin_pred_error=compute_classification_orient(talpha,orient_pred)
        
        bin_pred_error_list.append(bin_pred_error.cpu())
        
        p_alpha=get_alpha(orient_pred.cpu().data.numpy())
        
        theta_pred=get_theta_pred(roi_pred,selected_calib,center_pred,depth_pred,selected_shape,p_alpha,default_dims_tensor,pred_dim,selected_path)
        
        theta_target=labels[:, 11:12]*2*np.pi
        theta_target=theta_target.view(-1).cpu().numpy()
        theta_OS,_=get_orientation_score(np.array(theta_pred),theta_target)
        
        
        
        
        for idx_pred in range(len(center_pred)):
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
            
            # talpha_mod=torch.tensor([talpha[idx_pred,0:1]-2*np.pi,talpha[idx_pred,0:1]+2*np.pi,talpha[idx_pred,0:1]])
            # talpha_mod=talpha_mod[torch.min(abs(talpha_mod),0).indices].item()
            
            # mean_abs,_=get_orientation_score(p_alpha[idx_pred],talpha_mod)
            mean_abs,_=get_orientation_score(theta_pred[idx_pred],theta_target[idx_pred])

            orient_abs_err.append(torch.tensor(mean_abs))
            
            try:
                orient_abs_err_dict[obj_cls].append(mean_abs)
            except:
                orient_abs_err_dict[obj_cls]=[mean_abs]
            
            
        return center_abs_err,depth_abs_err,dim_abs_err,orient_abs_err,dim_abs_err_dict,orient_abs_err_dict,depth_abs_err_dict,bin_pred_error_list,loss_list

def compute_mean_errors_and_print(stats,center_abs_err,depth_abs_err,dim_abs_err,
                                  orient_abs_err,dim_abs_err_dict,orient_abs_err_dict,depth_abs_err_dict,bin_pred_error_list,nc,names,seen):
    
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
    pf = '%30s' + '%10.3g' * 9  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1, dim_abs_err, orient_abs_err, depth_abs_err))
    with open("output.txt","a") as f:
        f.write(pf % ('all', seen, nt.sum(), mp, mr, map, mf1, dim_abs_err, orient_abs_err, depth_abs_err)+'\n')

    # Print results per class
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
                    if iou > iou_thres:
                        associated.append((bi.cpu().item()+len_target,i+len_pred))
                    
                    if iou > iou_thres and m[bi] not in [ind[1] for ind in detected]:  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append((i,m[bi].cpu().item()))
                        

                        # associated.append((bi.cpu().item()+len_target,i+len_pred))

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
        iou_thres=0.45,
        conf_thres=0.25,
        nms_thres=0.45,
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
    depths_dict={}
    loss_list=[]
    with open("data/KITTI/avg_shapes.json","r") as f:
        default_dims=json.load(f)
    default_dims_tensor=torch.zeros(len(default_dims),3,device=device)
    for class_idx in default_dims:
        default_dims_tensor[int(class_idx),:]=torch.tensor([shape for shape in default_dims[class_idx]])
    
    for batch_i, (imgs, targets, paths, shapes,calib,_,_) in enumerate(tqdm(dataloader)):
        
        if len(targets)==0:
            print("skipping empty target")
            continue
        input_targets,width,height,imgs,targets=prepare_data_for_foward_pass(targets,device,imgs)
        calib=calib.to(imgs.device)
        with torch.cuda.amp.autocast():
            output_roi,center_pred, depth_pred ,pred_dim,orient_pred= model(imgs,conf_thres=conf_thres, nms_thres=nms_thres,detect=True)  # inference and training outputs

        stats,seen,associated=compute_bbox_error(output_roi,targets,stats,width,height,iou_thres,seen)
        center_abs_err,depth_abs_err,dim_abs_err,orient_abs_err,dim_abs_err_dict, orient_abs_err_dict, depth_abs_err_dict,bin_pred_error_list,loss_list=compute_center_and_depth_errors(center_pred,depth_pred,pred_dim,
                                                                        orient_pred,center_abs_err,depth_abs_err,
                                                                        dim_abs_err,orient_abs_err,targets,(width,height),
                                                                        default_dims_tensor,dim_abs_err_dict, orient_abs_err_dict,
                                                                        depth_abs_err_dict,bin_pred_error_list,associated,loss_list,output_roi,calib,shapes,paths)
        depths_dict=store_all_depths(depth_pred,targets,depths_dict,associated)
    
    
    
    (mp, mr, map, mf1, center_abs_err,
     depth_abs_err,dim_abs_err,orient_abs_err), maps=compute_mean_errors_and_print(stats,center_abs_err,depth_abs_err,
                                                                                   dim_abs_err,orient_abs_err,dim_abs_err_dict,
                                                                                   orient_abs_err_dict,depth_abs_err_dict,bin_pred_error_list,nc,names,seen)
    with open("test_loss.txt","a") as f:
        loss_test=torch.mean(torch.tensor(loss_list)).item()
        f.write(str(loss_test)+'\n')
        
        
    with open("depth_errors.txt",'w') as f:
        f.write(('%30s' + '%20s' * 10) % ('Class', 'Images', 'Targets', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'scale_invariant', 'a1', 'a2', 'a3')+'\n')
        pf='%30s' + '%20.3g' * 10
        stats = [np.concatenate(x, 0) for x in list(zip(*stats))]
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
        
        all_pred_depth=[]
        all_target_depth=[]
        for key in depths_dict:
            all_pred_depth+=depths_dict[key][0]
            all_target_depth+=depths_dict[key][1]
        all_pred_depth=np.array(all_pred_depth)
        all_target_depth=np.array(all_target_depth)
        
        abs_rel, sq_rel, rmse, rmse_log, scale_invariant, a1, a2, a3=compute_depth_errors(all_target_depth*200, all_pred_depth*200)
        
        f.write(pf % ('all', seen, nt.sum(), abs_rel, sq_rel, rmse, rmse_log, scale_invariant, a1, a2, a3)+'\n')
        for key in depths_dict:
            abs_rel, sq_rel, rmse, rmse_log, scale_invariant, a1, a2, a3=compute_depth_errors(np.array(depths_dict[key][1])*200, np.array(depths_dict[key][0])*200)
            f.write(pf % (names[key], seen, nt[key], abs_rel, sq_rel, rmse, rmse_log, scale_invariant, a1, a2, a3)+'\n')
    return (mp, mr, map, mf1, center_abs_err,depth_abs_err,dim_abs_err,orient_abs_err), maps





if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-3dcent-KITTI.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/KITTI.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='/media/antoine/NVMe/best.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
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
