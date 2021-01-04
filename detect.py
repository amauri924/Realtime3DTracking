import argparse
import time
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *
import test
from matplotlib import patches
import matplotlib.pyplot as plt

def detect(
        cfg,
        data_cfg,
        weights,
        images='data/samples',  # input folder
        output='output',  # output folder
        fourcc='mp4v',
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
        save_txt=False,
        save_images=True,
        webcam=False
):
    device = torch_utils.select_device()
    torch.backends.cudnn.benchmark = True  # set False for reproducible results
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    if ONNX_EXPORT:
        s = (320, 192)  # (320, 192) or (416, 256) or (608, 352) onnx model image size (height, width)
        model = Darknet(cfg, s)
    else:
        model = Model(cfg,0,transfer=False)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
#    model.fuse()
    with open("data/3dcent-NS/avg_shapes.json","r") as f:
        default_dims=json.load(f)
    default_dims_tensor=torch.zeros(len(default_dims),3,device=device)
    for class_idx in default_dims:
        default_dims_tensor[int(class_idx),:]=torch.tensor([shape for shape in default_dims[class_idx]])
    # Eval mode
    model.to(device).eval()

    if ONNX_EXPORT:
        img = torch.zeros((1, 3, s[0], s[1]))
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    for i, (path, img, im0, vid_cap, calib) in enumerate(dataloader):
        t = time.time()
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        det,center_pred, depth_pred,pred_dim,orient_pred = model(img,conf_thres=conf_thres, nms_thres=nms_thres,detect=True)
        det=det[0]
        
        
        # fig = plt.figure(figsize=(16, 16))
        # plt.title('RGB')
        # plt.imshow(im0)
        
#        det = non_max_suppression(pred, conf_thres, nms_thres)[0]
        p_alpha=test.get_alpha(orient_pred.cpu().data.numpy())
        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            idxs=torch.max(det[:,5:],1).indices
            centers=center_pred
            # centers=[(center_pred[i,idxs[i]],center_pred[i,idxs[i]+1]) for i in range(len(idxs))]
            depth_pred=depth_pred*200
            
            # Print results to screen
            print('%gx%g ' % img.shape[2:], end='')  # print image size
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for j,(*xyxy, conf, cls_conf, cls) in enumerate(det):
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                #get 3d center pixel coord
                # x=xyxy[0]+centers[j][0]*(xyxy[2]-xyxy[0])
                # y=xyxy[1]+centers[j][1]*(xyxy[3]-xyxy[1])
                
                w_bbox=xyxy[2]-xyxy[0]
                h_bbox=xyxy[3]-xyxy[1]
                centerbbox_x=xyxy[0]+w_bbox/2
                centerbbox_y=xyxy[1]+h_bbox/2
                
                x=(centers[j][0]*w_bbox+centerbbox_x).item()
                y=(centers[j][1].item()*h_bbox+centerbbox_y).item()
                depth=depth_pred[j].item()
                
                homo_center=np.array([[x*depth],[y*depth],[depth],[1]])
                center_3d=np.linalg.inv(calib) @ homo_center
                
                alpha=p_alpha[j] if p_alpha[j]>0 else p_alpha[j]+2*np.pi
                
                
                theta=alpha+math.atan2(center_3d[0][0],center_3d[2][0])
                
                
                width,height,length=(default_dims_tensor[int(cls),:] - pred_dim[j,int(cls),:]*200).cpu().numpy()
                model_size=pred_dim
                
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
                pixel_points = calib@points_3dbbox_homo_cam_coord
                pixel_points = pixel_points[:2,:] / pixel_points[2,:]
                
                pixel_points[1,:]=np.clip(pixel_points[1,:], 0, im0.shape[0])
                pixel_points[0,:]=np.clip(pixel_points[0,:], 0, im0.shape[1])
                
                bbox_2d = np.zeros((8, 2))
                for k in range(points_3dbbox_homo_cam_coord.shape[1]):
                    
                    bbox_2d[k, :] = pixel_points[:,k]
                    
                # pol1 = patches.Polygon(bbox_2d[(0, 1, 3, 2), :], closed=True, linewidth=1, edgecolor='g', facecolor='none')  # fixed x
                # pol2 = patches.Polygon(bbox_2d[(4, 5, 7, 6), :], closed=True, linewidth=1, edgecolor='g', facecolor='none')  # fixed x
                # pol3 = patches.Polygon(bbox_2d[(0, 2, 6, 4), :], closed=True, linewidth=1, edgecolor='g', facecolor='none')  # fixed z
                # pol4 = patches.Polygon(bbox_2d[(1, 3, 7, 5), :], closed=True, linewidth=1, edgecolor='g', facecolor='none')  # fixed z
                # pol5 = patches.Polygon(bbox_2d[(0, 1, 5, 4), :], closed=True, linewidth=1, edgecolor='g', facecolor='none')  # fixed y
                # pol6 = patches.Polygon(bbox_2d[(2, 3, 7, 6), :], closed=True, linewidth=1, edgecolor='g', facecolor='none')  # fixed y
                # plt.gca().add_patch(pol1)
                # plt.gca().add_patch(pol2)
                # plt.gca().add_patch(pol3)
                # plt.gca().add_patch(pol4)
                # plt.gca().add_patch(pol5)
                # plt.gca().add_patch(pol6)
                
                # fig.savefig(save_path,dpi=200)
                bbox_2d=bbox_2d.round().astype(np.int32)
                # bbox_2d=bbox_2d.reshape(-1,1,2)
                
                cv2.polylines(im0,[bbox_2d[(0, 1, 3, 2), :].reshape(-1,1,2)],False,colors[int(cls)])
                cv2.polylines(im0,[bbox_2d[(4, 5, 7, 6), :].reshape(-1,1,2)],False,colors[int(cls)])
                cv2.polylines(im0,[bbox_2d[(0, 2, 6, 4), :].reshape(-1,1,2)],False,colors[int(cls)])
                cv2.polylines(im0,[bbox_2d[(1, 3, 7, 5), :].reshape(-1,1,2)],False,colors[int(cls)])
                cv2.polylines(im0,[bbox_2d[(0, 1, 5, 4), :].reshape(-1,1,2)],False,colors[int(cls)])
                cv2.polylines(im0,[bbox_2d[(2, 3, 7, 6), :].reshape(-1,1,2)],False,colors[int(cls)])
                
                orient_object = np.array([
                [0, y_max, 0,1],
                [0, y_max, 2,1],
                ]).T
                
                orient_pixels=calib@K_model_to_cam@orient_object
                orient_pixels = (orient_pixels[:2,:] / orient_pixels[2,:])
                
                orient_pixels[1,:]=np.clip(orient_pixels[1,:], 0, im0.shape[0])
                orient_pixels[0,:]=np.clip(orient_pixels[0,:], 0, im0.shape[1])
                orient_pixels=orient_pixels.round().astype(np.int32).T
                
                cv2.arrowedLine(im0,tuple(orient_pixels[0, :]),tuple(orient_pixels[1, :]),color=colors[int(cls)],thickness=2)
                
                cv2.circle(im0,(round(x),round(y)),3,color=colors[int(cls)])
                
                
                
                # Add bbox to the image
                label = '%s %.2f %.1fm' % (classes[int(cls)], conf, depth)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)],line_thickness=2)

        print('Done. (%.3fs)' % (time.time() - t))

        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)

        if save_images:  # Save image with detections
            if dataloader.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
                vid_writer.write(im0)

    if save_images:
        print('Results saved to %s' % os.getcwd() + os.sep + output)
        if platform == 'darwin':  # macos
            os.system('open ' + output + ' ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/3dcent-COCO.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/CRIANN/latest.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/3dcent-NS/test.txt', help='path to images')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.2, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='fourcc output video codec (verify ffmpeg support)')
    parser.add_argument('--output', type=str, default='output', help='specifies the output path for images and videos')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt.cfg,
               opt.data_cfg,
               opt.weights,
               images=opt.images,
               img_size=opt.img_size,
               conf_thres=opt.conf_thres,
               nms_thres=opt.nms_thres,
               fourcc=opt.fourcc,
               output=opt.output)
