import argparse
#import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *
import cv2

def detection(
        cfg,
        data_cfg,
        weights=None,
        img_size=416,
        conf_thres=0.1,
        nms_thres=0.5,
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

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device
    depth_bin=[1.662120213105634647e+01,
               3.226488398501740562e+01,5.094282679935143676e+01,
               7.380026966365950614e+01,1.111202708277208160e+02] #Create the depth bin centers. The bin width is 24m
    # Configure run
    model.transfer=False
    data_cfg = parse_data_cfg(data_cfg)
    nc = int(data_cfg['classes'])  # number of classes
    test_path = data_cfg['valid']  # path to test images
    classes = load_classes(data_cfg['names'])  # class names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size, 1)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=0,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    model.eval()
    print(('%30s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1'))
    for batch_i, (imgs, targets, paths, shapes) in enumerate(dataloader):

        if paths[0].endswith("scene757_sample3.png"):
            print("ah")


        imgs = imgs.to(device)
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        # Plot images with bounding boxes
#        if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
#            plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.jpg')

        # Run model
        output, center_pred_list, depth_pred_list = model(imgs,conf_thres=conf_thres, nms_thres=nms_thres)  # inference and training outputs

        

        # Run NMS
#        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)

        # Statistics per image
        for si, pred in enumerate(output):
            if pred is None:
                continue
            labels = targets[targets[:, 0] == si, 1:]
            # target boxes
            tbox = xywh2xyxy(labels[:, 1:5])
            tbox[:, [0, 2]] *= width
            tbox[:, [1, 3]] *= height
            
            
            save_path = str(Path("output") / Path(paths[si]).name)
            im0=cv2.imread(paths[si])
            
            pred[:, :4] = scale_coords(imgs.shape[2:], pred[:, :4], im0.shape).round()
            
            center_pred=center_pred_list[si]
            depth_pred=depth_pred_list[si]

            for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):
                depth_bin_idx=torch.max(depth_pred[i,int(pcls),:,1],0)[1]
                pdepth=depth_pred[i,int(pcls),depth_bin_idx,0]*200+depth_bin[depth_bin_idx]
                # Add bbox to the image
                label = '%s %.2f Distance:%.2f' % (classes[int(pcls)], pcls_conf,pdepth)
                plot_one_box(pbox, im0, label=label, color=colors[int(pcls)])
            cv2.imwrite(save_path, im0)
                
#                pcenter=center_pred[i]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-3dcent-NS.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/3dcent-NS.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/best_1.8.pt', help='path to weights file')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.3, help='iou threshold for non-maximum suppression')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        mAP = detection(opt.cfg,
                   opt.data_cfg,
                   opt.weights,
                   opt.img_size,
                   opt.conf_thres,
                   opt.nms_thres)
