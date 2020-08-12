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
    for batch_i, (imgs, targets, paths, shapes,_) in enumerate(dataloader):
        input_targets=targets.numpy()

        
        targets = targets.to(device)
        imgs = imgs.to(device)
        _, _, height, width = imgs.shape  # batch size, channels, height, width
        
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
        
        
        center_pred, depth_pred = model(imgs,conf_thres=conf_thres, nms_thres=nms_thres,testing=True,targets=input_targets)  # inference and training outputs



        # Run NMS
#        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)

        # Statistics per image
        labels=targets[:,1:]
        tcls=labels[:,0]

        tbox = xywh2xyxy(labels[:, 1:5])
        tbox[:, [0, 2]] *= width
        tbox[:, [1, 3]] *= height
        
        tcent=labels[:, 5:7]
        tdepth=labels[:,7]

        tcent[:, 0] *= width
        tcent[:, 1] *= height

        bceloss=nn.BCEWithLogitsLoss(reduction='sum')
        depth_bin=[100] #Create the depth bin centers. The bin width is 24m
        for idx_pred in range(len(center_pred)):
            target_center=tcent[idx_pred]
            predicted_center=center_pred[idx_pred]
            
            gt_depth=tdepth[idx_pred]
            predicted_depth=depth_pred[idx_pred]
            
            
            obj_cls=int(tcls[idx_pred])
            predicted_center=predicted_center[obj_cls:obj_cls+2]
            

            
            w_bbox=tbox[idx_pred][2].cpu().item()-tbox[idx_pred][0].cpu().item()
            h_bbox=tbox[idx_pred][3].cpu().item()-tbox[idx_pred][1].cpu().item()
            centerbbox_x=tbox[idx_pred][0].cpu().item()+w_bbox/2
            centerbbox_y=tbox[idx_pred][1].cpu().item()+h_bbox/2
            
            predicted_center[0]=predicted_center[0]*w_bbox+centerbbox_x
            predicted_center[1]=predicted_center[1]*h_bbox+centerbbox_y
            

            center_abs_err.append(torch.mean(torch.tensor([abs(abs(predicted_center[0]-target_center[0])/target_center[0]),abs(abs(predicted_center[1]-target_center[1])/target_center[1])])))
            depth_abs_err.append(abs(abs(predicted_depth-gt_depth)/(gt_depth+0.00001)))

            
            


        
        



    # Return results
    maps = np.zeros(nc) + map
    if len(center_abs_err)>0:
        center_abs_err=torch.mean(torch.tensor(center_abs_err)[torch.isfinite(torch.tensor(center_abs_err))]).cpu().item()
        depth_abs_err=torch.mean(torch.tensor(depth_abs_err)[torch.isfinite(torch.tensor(depth_abs_err))]).cpu().item()
        if math.isnan(depth_abs_err):
            print("nan")

    else:
        center_abs_err=0
        depth_abs_err=0
        mean_real_depth_abs_err=0
        loss_dconf=0
    return (center_abs_err,depth_abs_err,depth_abs_err,depth_abs_err)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=12, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-3dcent-NS.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/3dcent-NS.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/best_1.1.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
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
