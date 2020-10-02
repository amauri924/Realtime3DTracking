
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:41:18 2020

@author: antoine
"""

import glob
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image, ExifTags

from utils.utils import xyxy2xywh, xywh2xyxy


img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif']
vid_formats = ['.mov', '.avi', '.mp4']


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        None

    return s

def parse_data_cfg(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=416, batch_size=16, augment=False, rect=True, image_weights=False):
        with open(path, 'r') as f:
            img_files = f.read().splitlines()
            self.img_files = [x for x in img_files if os.path.splitext(x)[-1].lower() in img_formats]

        n = len(self.img_files)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        assert n > 0, 'No images found in %s' % path

        self.n = n
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.image_weights = image_weights
        self.rect = False if image_weights else rect

        # Define labels
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            # Read image shapes
            sp = 'data' + os.sep + path.replace('.txt', '.shapes').split(os.sep)[-1]  # shapefile path
            if not os.path.exists(sp):  # read shapes using PIL and write shapefile for next time (faster)
                s = [exif_size(Image.open(f)) for f in self.img_files]
                np.savetxt(sp, s, fmt='%g')

            with open(sp, 'r') as f:  # read existing shapefile
                s = np.array([x.split() for x in f.read().splitlines()], dtype=np.float64)
                assert len(s) == n, 'Shapefile error. Please delete %s and rerun' % sp  # TODO: auto-delete shapefile

            # Sort by aspect ratio
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            i = ar.argsort()
            self.img_files = [self.img_files[i] for i in i]
            self.label_files = [self.label_files[i] for i in i]
            ar = ar[i]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32.).astype(np.int) * 32

        # Preload labels (required for weighted CE training)
        self.imgs = [None] * n
        self.labels = [None] * n
        preload_labels = False
        if preload_labels:
            self.labels = [np.zeros((0, 5))] * n
            iter = self.label_files if n > 10 else self.label_files
            extract_bounding_boxes = False
            for i, file in enumerate(iter):
                try:
                    with open(file, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                        if l.shape[0]:
                            assert l.shape[1] == 5, '> 5 label columns: %s' % file
                            assert (l >= 0).all(), 'negative labels: %s' % file
                            assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                            self.labels[i] = l

                            # Extract object detection boxes for a second stage classifier
                            if extract_bounding_boxes:
                                p = Path(self.img_files[i])
                                img = cv2.imread(str(p))
                                h, w, _ = img.shape
                                for j, x in enumerate(l):
                                    f = '%s%sclassification%s%g_%g_%s' % (
                                        p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                                    if not os.path.exists(Path(f).parent):
                                        os.makedirs(Path(f).parent)  # make new output folder
                                    box = xywh2xyxy(x[1:].reshape(-1, 4)).ravel()
                                    box = np.clip(box, 0, 1)  # clip boxes outside of image
                                    result = cv2.imwrite(f, img[int(box[1] * h):int(box[3] * h),
                                                            int(box[0] * w):int(box[2] * w)])
                                    if not result:
                                        print('stop')
                except:
                    pass  # print('Warning: missing labels for %s' % self.img_files[i])  # missing label file
            assert len(np.concatenate(self.labels, 0)) > 0, 'No labels found. Incorrect label paths provided.'

        # Detect corrupted images https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io  # conda install -c conda-forge scikit-image
            for file in self.img_files:
                try:
                    _ = io.imread(file)
                except:
                    print('Corrupted image detected: %s' % file)

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        img_path = self.img_files[index]
        label_path = self.label_files[index]
        calib_path=self.img_files[index].split('.')[0]+'.npy'
        calib=np.load(calib_path).astype(np.float32)

        # Load image
        img = self.imgs[index]
        if img is None:
            img = cv2.imread(img_path)  # BGR
            h,w,_=img.shape
            assert img is not None, 'File Not Found ' + img_path
            if self.n < 1001:
                self.imgs[index] = img  # cache image into memory

        # Load labels
        labels = []
        if os.path.isfile(label_path):
            x = self.labels[index]
            if x is None:  # labels not preloaded
                with open(label_path, 'r') as f:
                    x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                    self.labels[index] = x  # save for next time
            
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2)
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2)
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2)
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2)
                
                labels[:, 5] = (w * x[:, 5])
                labels[:, 6] = (h * x[:, 6])


        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:,6]/=img.shape[0]
            labels[:,5]/=img.shape[1]
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        labels_out = torch.zeros((nL, 6+3))

        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        #Resize image
        if h!=720 or w!=1280:
            img=cv2.resize(img,(1280,720))

        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return torch.from_numpy(img), labels_out, img_path, (h, w),torch.tensor(calib)

    @staticmethod
    def collate_fn(batch):
        img, label, path, hw,calib = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, hw, torch.stack(calib,0)
