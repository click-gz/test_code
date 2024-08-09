#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 11:06:19 2021

@author: leeh43
"""
import SimpleITK as sitk
from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from networks.UXNet_3D.network_backbone import UXNET
# from networks.UXNet_3D.network_cube1_backbone import UXNET
from monai.networks.nets import UNETR, SwinUNETR
from networks.nnFormer.nnFormer_seg import nnFormer
# from networks.nnFormer.nnFormer_cube_seg import nnFormer
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.metrics import DiceMetric, compute_hausdorff_distance
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch

import torch
from torch.utils.tensorboard import SummaryWriter
from load_datasets_transforms import data_loader, data_transforms

import os, glob
import numpy as np
from tqdm import tqdm
import argparse
from surface_distance import *
def hd95(f, m):
    hd95 = 0
    count = 0
    for i in range(0, 2):
        if ((f == i).sum() == 0) or ((m == i).sum() == 0):
            continue
        # print(i)
        hd95 += compute_robust_hausdorff(compute_surface_distances((f == i), (m == i), np.ones(3)), 95.)
        count += 1
    hd95 /= count
    print(hd95, count)
    return hd95


def calculate_boundary_f1(mask_true, mask_pred):
    # Calculate true positive, false positive, and false negative boundary pixels
    true_positive = np.sum(np.logical_and(mask_true, mask_pred))
    false_positive = np.sum(np.logical_and(np.logical_not(mask_true), mask_pred))
    false_negative = np.sum(np.logical_and(mask_true, np.logical_not(mask_pred)))

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
seg_files = glob.glob("/data1/gz/data/out1_seg/*/*")
hd = []
from sklearn.metrics import f1_score
f1 = []
# print(f1_score(y_true, y_pred, average='weighted'))
# print(f1_score(y_true, y_pred, average='macro'))
def iou_score(m, f):
    smooth = 1e-5
    iou=0
    count=0
    for i in range(0, 2):
        # if ((f == i).sum() == 0) or ((m == i).sum() == 0):
        #     continue
        intersection = ((f==i) & (m==i)).sum()
        union = ((f==i) |  (m==i)).sum()
        iou += (intersection + smooth) / (union + smooth)
        count += 1
    print(iou/count, count)
    return iou/count
iou = []
for f in seg_files:
    label = "/data1/gz/data/Lung250M/test_seg/" + f.split("/")[-2]+".nii.gz"
    label = sitk.GetArrayFromImage(sitk.ReadImage(label))
    seg = sitk.GetArrayFromImage(sitk.ReadImage(f))
    hdr = hd95(seg, label)
    f1r = calculate_boundary_f1(seg, label)
    iour = iou_score(seg, label)
    f1.append(f1r)
    hd.append(hdr)
    iou.append(iour)
    print(hdr, f1r)
print(np.mean(f1), np.std(f1))
print(np.mean(hd), np.std(hd))
print(np.mean(iou), np.std(iou))


