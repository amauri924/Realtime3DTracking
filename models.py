import os

import torch.nn.functional as F
import time
from utils.parse_config import *
from utils.utils import *
from torchvision.ops import roi_align
from torch.autograd import Variable


ONNX_EXPORT = False


def create_modules():
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    modules = nn.Sequential()
    modules.add_module('Yolov3',Yolov3())
    modules.add_module('roi_align',RoiAlign())
    module_list.append(modules)
    return modules

class Yolov3(nn.Module):
    def __init__(self, cfg, img_size=(416, 416)):
        super(Yolov3,self).__init__()
        self.module_defs = parse_model_cfg(cfg)
        self.module_defs[0]['cfg'] = cfg
        self.module_defs[0]['height'] = img_size
        self.hyperparams, self.module_list = self.init_module(self.module_defs)
        self.yolo_layers = get_yolo_layers(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
    
    
    def forward(self, x, var=None):
        img_size = max(x.shape[-2:])
        layer_outputs = []
        output = []
        detection_time=time.time()
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif mtype == 'yolo':
                x = module[0](x, img_size)
                output.append(x)
            layer_outputs.append(x)
            if i==74:
                features=x
            
        time_detection=time.time()-detection_time
        p,io_orig = list(zip(*output))  # inference output, training output
#        io=torch.cat(io, 1)
        del x,layer_outputs
        return p,features,io_orig
    
    
    
    
    def init_module(self,module_defs):
        """
        Constructs module list of layer blocks from module configuration in module_defs
        """
        hyperparams = module_defs.pop(0)
        output_filters = [int(hyperparams['channels'])]
        module_list = nn.ModuleList()
        yolo_index = -1
    
        for i, module_def in enumerate(module_defs):
            modules = nn.Sequential()
    
            if module_def['type'] == 'convolutional':
                bn = int(module_def['batch_normalize'])
                filters = int(module_def['filters'])
                kernel_size = int(module_def['size'])
                pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
                modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                            out_channels=filters,
                                                            kernel_size=kernel_size,
                                                            stride=int(module_def['stride']),
                                                            padding=pad,
                                                            bias=not bn))
                if bn:
                    modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
                if module_def['activation'] == 'leaky':
                    modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1, inplace=True))
    
            elif module_def['type'] == 'maxpool':
                kernel_size = int(module_def['size'])
                stride = int(module_def['stride'])
                if kernel_size == 2 and stride == 1:
                    modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
                maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
                modules.add_module('maxpool_%d' % i, maxpool)
    
            elif module_def['type'] == 'upsample':
                upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')
                modules.add_module('upsample_%d' % i, upsample)
    
            elif module_def['type'] == 'route':
                layers = [int(x) for x in module_def['layers'].split(',')]
                filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
                modules.add_module('route_%d' % i, EmptyLayer())
    
            elif module_def['type'] == 'shortcut':
                filters = output_filters[int(module_def['from'])]
                modules.add_module('shortcut_%d' % i, EmptyLayer())
    
            elif module_def['type'] == 'yolo':
                yolo_index += 1
                anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
                # Extract anchors
                anchors = [float(x) for x in module_def['anchors'].split(',')]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in anchor_idxs]
                nc = int(module_def['classes'])  # number of classes
                img_size = hyperparams['height']
                # Define detection layer
                modules.add_module('yolo_%d' % i, YOLOLayer(anchors, nc, img_size, yolo_index))
    
            # Register module list and number of output filters
            module_list.append(modules)
            output_filters.append(filters)
            
        return hyperparams, module_list

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride,
                               bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               # change
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)
#        del residual

        return x


class ConvBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)  # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x



class downsample(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(downsample, self).__init__()
        self.modules_list=self.init_modules(in_channel,out_channel)
    
    def forward(self,x):
        x=self.modules_list(x)
        return x
    def init_modules(self,in_channel,out_channel):
        modules=nn.Sequential(nn.Conv2d(in_channel,out_channel, kernel_size=(1, 1), stride=(2, 2), bias=False),
                              nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        return modules

class Top_layer(nn.Module):
    def __init__(self):
        super(Top_layer, self).__init__()
        self.module_list=self.init_mod()

    def forward(self,x):
        x=self.module_list(x).mean(3).mean(2)
        return x
        
    def init_mod(self):
        modules=nn.Sequential(Bottleneck(1024,512,stride=2,downsample=downsample(1024,2048)),
                Bottleneck(2048,512,stride=1),Bottleneck(2048,512,stride=1))
                              
        return modules

class Depth_Layer(nn.Module):
    def __init__(self,nc):
        super(Depth_Layer, self).__init__()
        self.nc=nc
        self.convblock=ConvBlock(1024,2048)
        self.lin=nn.Linear(in_features=2048, out_features=1*self.nc, bias=True)
        self.relu= nn.ReLU(inplace=True)
        

    def forward(self,x):
        x=self.convblock(x).mean(3).mean(2)
        x=self.lin(x)
        x=self.relu(x)
        return x
        
    def init_mod(self):
        modules=nn.Sequential(ConvBlock(1024,2048),nn.Linear(in_features=2048, out_features=1*self.nc, bias=True))
        return modules

class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x

class RoiAlign(nn.Module):
    def __init__(self):
        super(RoiAlign, self).__init__()
        

    def forward(self,features,filtered_roi,batch_idxs=0):

        if features.type()=='torch.cuda.HalfTensor':
            box=torch.zeros([filtered_roi.shape[0], 5], dtype=torch.float16).to(features.device)
        else:
            box=torch.zeros([filtered_roi.shape[0], 5], dtype=torch.float32).to(features.device)
        box[:,1:] = filtered_roi[:, :4].detach()
#        if batch_idxs!=torch.tensor([]):
        if self.training:
            box[:,0]=batch_idxs
        pooled_features=roi_align(features, box, (7,7), spatial_scale=1.0/32.0)
        
        del box
        return pooled_features
    
    



class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints

        if ONNX_EXPORT:  # grids must be computed in __init__
            stride = [32, 16, 8][yolo_index]  # stride of this layer
            nx = int(img_size[1] / stride)  # number x grid points
            ny = int(img_size[0] / stride)  # number y grid points
            create_grids(self, max(img_size), (nx, ny))

    def forward(self, p, img_size, var=None):
        if ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, ny, nx = p.shape[0], p.shape[-2], p.shape[-1]
            if (self.nx, self.ny) != (nx, ny):
                create_grids(self, img_size, (nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.nc + 5, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

#        else:  # inference
        io = p.clone().detach()  # inference output
        io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
        io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
        # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
        io[..., 4:] = torch.sigmoid(io[..., 4:])  # p_conf, p_cls
        # io[..., 5:] = F.softmax(io[..., 5:], dim=4)  # p_cls
        io[..., :4] *= self.stride
        if self.nc == 1:
            io[..., 5] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235

        # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
        return p,io


class center_pred(nn.Module):
    def __init__(self,nc):
        super(center_pred, self).__init__()
        self.nc=nc #num classes
        self.modules_list=nn.Sequential(nn.Linear(in_features=2048, out_features=2*self.nc, bias=True))

    
    def forward(self,x):
        x=self.modules_list(x)
        return x

class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg,hyp, img_size=(416, 416)):
        super(Darknet, self).__init__()

        self.Yolov3=Yolov3(cfg,img_size)
        self.nc=int(self.Yolov3.module_defs[-1]['classes']) #Get num classes
        self.featurePooling=RoiAlign()
        
        self.top_layer=Top_layer()
        self.depth_pred=Depth_Layer(self.nc)
        self.center_prediction=center_pred(self.nc)
        
        self.hyp=hyp
        self._init_weights()

    def forward(self, x, var=None,targets=None,conf_thres=0,nms_thres=0):

        
        if not self.training:
            _ ,features,io_orig=  self.Yolov3(x) # inference output, training output
            io=[]
            for line in io_orig:
                line=line.view(io_orig[0].shape[0], -1, 5 + self.nc)
                io.append(line)
            rois=torch.cat(io,1)
            time_NMS=time.time()
            rois = non_max_suppression(rois, conf_thres=conf_thres, nms_thres=nms_thres)
            time_NMS=time.time()-time_NMS
            center_pred_list=[]
            depth_pred_list=[]
            for roi in rois:
                if roi is None:
                    center_pred_list.append(None)
                    depth_pred_list.append(None)
                    continue
                pooled_features=self.featurePooling(features, roi)
                depth_pred=self.depth_pred(pooled_features)
                pooled_features=self.top_layer(pooled_features) # Run the final layers 
                center_pred=self.center_prediction(pooled_features)/100 # Run the 3D prediction
                center_pred_list.append(center_pred)
                depth_pred_list.append(depth_pred)
                del pooled_features
            del features
            return rois,center_pred_list,depth_pred_list
        else:
            p ,features,_=  self.Yolov3(x) # inference output, training output
            targets=torch.from_numpy(targets)
            targets=targets.to(features.device)
            device_id=int(str(x.device)[-1])
            
            #Compute 3D center using GT bbox
            #For multi-gpu
            new_idxs=targets[:,0]-device_id*x.shape[0]
            mask_1=new_idxs<x.shape[0]
            mask_2=new_idxs>-1
            mask_3=mask_1&mask_2
            targets=targets[mask_3]
            
            roi=targets[:,2:6]
            b=new_idxs[mask_3]
            pooled_features=self.featurePooling(features, roi,b)
            depth_pred=self.depth_pred(pooled_features)
            del b,roi
            top_layer=self.top_layer(pooled_features) # Run the final layers 
            center_pred=self.center_prediction(top_layer)/100 # Run the 3D prediction
            
            
            return p,center_pred,depth_pred
        
        
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()
        normal_init(self.center_prediction.modules_list[0], 0, 0.001)


    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            for i, b in enumerate(a):
                if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                    # fuse this bn layer with the previous conv2d layer
                    conv = a[i - 1]
                    fused = torch_utils.fuse_conv_and_bn(conv, b)
                    a = nn.Sequential(fused, *list(a.children())[i + 1:])
                    break
            fused_list.append(a)
        self.module_list = fused_list
        # model_info(self)  # yolov3-spp reduced from 225 to 152 layers


def get_yolo_layers(model):
    a = [module_def['type'] == 'yolo' for module_def in model.module_defs]
    return [i for i, x in enumerate(a) if x]  # [82, 94, 106] for yolov3


def create_grids(self, img_size=416, ng=(13, 13), device='cpu'):
    nx, ny = ng  # x and y grid size
    self.img_size = img_size
    self.stride = img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).float().view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    weights_file = weights.split(os.sep)[-1]

    # Try to download weights if not available locally
    if not os.path.isfile(weights):
        try:
            url = 'https://pjreddie.com/media/files/' + weights_file
            print('Downloading ' + url + ' to ' + weights)
            os.system('curl ' + url + ' -o ' + weights)
            import requests
            r = requests.get(url)

        except IOError:
            print(weights + ' not found.\nTry https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI')

    # Establish cutoffs
    if weights_file == 'darknet53.conv.74':
        cutoff = 75
    elif weights_file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.Yolov3.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.Yolov3.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.Yolov3.module_defs[:cutoff], self.Yolov3.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

    return cutoff


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)

    # Load weights and save
    if weights.endswith('.pt'):  # if PyTorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        save_weights(model, path='converted.weights', cutoff=-1)
        print("Success: converted '%s' to 'converted.weights'" % weights)

    elif weights.endswith('.weights'):  # darknet format
        _ = load_darknet_weights(model, weights)

        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}

        torch.save(chkpt, 'converted.pt')
        print("Success: converted '%s' to 'converted.pt'" % weights)

    else:
        print('Error: extension not supported.')
