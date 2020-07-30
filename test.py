import argparse
#import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *


def test(
        cfg,
        data_cfg,
        weights=None,
        batch_size=16,
        img_size=416,
        iou_thres=0.5,
        conf_thres=0.1,
        nms_thres=0.5,
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

    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size, batch_size)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=28,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    coco91class = coco80_to_coco91_class()
    print(('%30s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1'))
    loss, p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    center_abs_err=[]
    depth_abs_err=[]
    loss_dconf=[]
    real_depth_abs_err=[]
    for batch_i, (imgs, targets, paths, shapes) in enumerate(dataloader):

        
        targets = targets.to(device)
        imgs = imgs.to(device)
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        # Plot images with bounding boxes
#        if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
#            plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.jpg')
#        if imgs.shape[0]!=batch_size:
#            continue  # inference and training outputs
        # Run model
        output, center_pred_list, depth_pred_list = model(imgs,conf_thres=conf_thres, nms_thres=nms_thres,testing=True)  # inference and training outputs



        # Run NMS
#        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)

        # Statistics per image
        for si, pred in enumerate(output):
            center_pred=center_pred_list[si]
            depth_pred=depth_pred_list[si]
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if len(pred)==0:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for di, d in enumerate(pred):
                    jdict.append({
                        'image_id': image_id,
                        'category_id': coco91class[int(d[6])],
                        'bbox': [float3(x) for x in box[di]],
                        'score': float(d[4])
                    })

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            associated_target = []
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= width
                tbox[:, [1, 3]] *= height
                
                tcent=labels[:, 5:7]
                tdepth=labels[:,7]*200

                tcent[:, 0] *= width
                tcent[:, 1] *= height

                
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
                
                #Compute 3D center error + depth error
                if len(detected)>0:
                    bceloss=nn.BCEWithLogitsLoss(reduction='sum')
                    depth_bin=[1.662120213105634647e+01,
                   3.226488398501740562e+01,
                   5.094282679935143676e+01,
                   7.380026966365950614e+01,
                   1.111202708277208160e+02
                   ] #Create the depth bin centers. The bin width is 24m
                    for idx_pred,idx_target in detected:
                        target_center=tcent[idx_target]
                        predicted_center=center_pred[idx_pred]
                        
                        gt_depth=tdepth[idx_target]
                        predicted_depth=depth_pred[idx_pred]
                        
                        dtarget=torch.zeros(len(depth_bin)).to(gt_depth.device)
                        for i,d_bin in enumerate(depth_bin):
                            dtarget[i]=gt_depth-d_bin
                        tconf_depth=(abs(dtarget)<24).type(torch.float)
                        
                        obj_cls=int(tcls[idx_target])
                        predicted_center=predicted_center[obj_cls:obj_cls+2]
                        
                        depth_pconf=predicted_depth[obj_cls,:,1]
                        p_depth=predicted_depth[obj_cls:obj_cls+1,torch.min(abs(dtarget),0)[1],0]*200+depth_bin[torch.min(abs(dtarget),0)[1]] #use GT to get the bin that is closer to the gt depth
                        real_p_depth=predicted_depth[obj_cls:obj_cls+1,torch.max(depth_pconf,0)[1],0]*200+depth_bin[torch.max(depth_pconf,0)[1]]
                        
                        w_bbox=pred[idx_pred][2].cpu().item()-pred[idx_pred][0].cpu().item()
                        h_bbox=pred[idx_pred][3].cpu().item()-pred[idx_pred][1].cpu().item()
                        centerbbox_x=pred[idx_pred][0].cpu().item()+w_bbox/2
                        centerbbox_y=pred[idx_pred][1].cpu().item()+h_bbox/2
                        
                        predicted_center[0]=predicted_center[0]*w_bbox+centerbbox_x
                        predicted_center[1]=predicted_center[1]*h_bbox+centerbbox_y
                        
                        loss_dconf.append(bceloss(depth_pconf,tconf_depth)) #compute BCE loss for the conf
                        center_abs_err.append(torch.mean(torch.tensor([abs(abs(predicted_center[0]-target_center[0])/target_center[0]),abs(abs(predicted_center[1]-target_center[1])/target_center[1])])))
                        depth_abs_err.append(abs(abs(p_depth[0]-gt_depth)/(p_depth[0]+0.00001)))
                        real_depth_abs_err.append(abs(abs(real_p_depth[0]-gt_depth)/(real_p_depth[0]+0.00001)))
                
                
            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

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
    if len(center_abs_err)>0:
        center_abs_err=torch.mean(torch.tensor(center_abs_err)[torch.isfinite(torch.tensor(center_abs_err))]).cpu().item()
        depth_abs_err=torch.mean(torch.tensor(depth_abs_err)[torch.isfinite(torch.tensor(depth_abs_err))]).cpu().item()
        mean_real_depth_abs_err=torch.mean(torch.tensor(real_depth_abs_err)[torch.isfinite(torch.tensor(real_depth_abs_err))]).cpu().item()
        if math.isnan(mean_real_depth_abs_err):
            print("nan")
        loss_dconf=torch.mean(torch.tensor(loss_dconf)).cpu().item()
    else:
        center_abs_err=0
        depth_abs_err=0
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, loss / len(dataloader), center_abs_err,loss_dconf,depth_abs_err,mean_real_depth_abs_err), maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=12, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-3dcent-NS.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/3dcent-NS.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/best_1.8.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
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
