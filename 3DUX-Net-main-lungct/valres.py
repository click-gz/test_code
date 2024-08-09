#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 11:06:19 2021

@author: leeh43
"""
import SimpleITK as sitk
from monai.utils import set_determinism
from monai.transforms import AsDiscrete
# from networks.UXNet_3D.network_backbone import UXNET
from networks.UXNet_3D.network_cube_backbone import UXNET
from monai.networks.nets import UNETR, SwinUNETR
from networks.SwinUNETR.swunetr1 import SwinUNETR
from networks.UNETR.unetr import UNETR
# from networks.nnFormer.nnFormer_Wcube_seg1 import nnFormer
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.metrics import DiceMetric, compute_hausdorff_distance
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch

import torch
from torch.utils.tensorboard import SummaryWriter
from load_datasets_transforms import data_loader, data_transforms

import os
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='3D UX-Net hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='/data1/gz/data/Lung250M', required=False, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='out_out', required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='lung', required=False, help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')


## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='3DUXNET',
                    help='Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET}')
parser.add_argument('--mode', type=str, default='train_seg', help='Training or testing mode')
parser.add_argument('--pretrain', default=False, help='Have pretrained weights or not')
parser.add_argument('--pretrained_weights', default='', help='Path of pretrained weights')
parser.add_argument('--batch_size', type=int, default='1', help='Batch size for subject input')
parser.add_argument('--crop_sample', type=int, default='2', help='Number of cropped sub-volumes for each subject')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer types: Adam / AdamW')
parser.add_argument('--max_iter', type=int, default=40000, help='Maximum iteration steps for training')
parser.add_argument('--eval_step', type=int, default=500, help='Per steps to perform validation')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print('Used GPU: {}'.format(args.gpu))

train_samples, valid_samples, out_classes = data_loader(args)

val_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(valid_samples['images'], valid_samples['labels'])
]

set_determinism(seed=0)

train_transforms, val_transforms = data_transforms(args)

## Train Pytorch Data Loader and Caching
print('Start caching datasets!')

## Valid Pytorch Data Loader and Caching
val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=args.cache_rate / 10, num_workers=args.num_workers)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)

## Load Networks
device = torch.device("cuda:0")
if args.network == '3DUXNET':
    model = UXNET(
        in_chans=1,
        out_chans=out_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    ).to(device)
elif args.network == 'SwinUNETR':
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=out_classes,
        feature_size=48,
        use_checkpoint=False,
    ).to(device)
elif args.network == 'nnFormer':
    model = nnFormer(input_channels=1, num_classes=out_classes).to(device)

elif args.network == 'UNETR':
    model = UNETR(
        in_channels=1,
        out_channels=out_classes,
        img_size=(96, 96, 96),
        feature_size=6,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)
elif args.network == 'TransBTS':
    _, model = TransBTS(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
    model = model.to(device)

print('Chosen Network Architecture: {}'.format(args.network))

if args.pretrain == 'True':
    print('Pretrained weight is found! Start to load weight from: {}'.format(args.pretrained_weights))
    model.load_state_dict(torch.load(args.pretrained_weights))

## Define Loss function and optimizer
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
print('Loss for training: {}'.format('DiceCELoss'))
if args.optim == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
elif args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
print('Optimizer for training: {}, learning rate: {}'.format(args.optim, args.lr))
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1000)


root_dir = os.path.join(args.output)
if os.path.exists(root_dir) == False:
    os.makedirs(root_dir)

t_dir = os.path.join(root_dir, 'tensorboard')
if os.path.exists(t_dir) == False:
    os.makedirs(t_dir)
writer = SummaryWriter(log_dir=t_dir)


def validation(epoch_iterator_val):
    # model_feat.eval()
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            # val_outputs = model(val_inputs)
            # print(val_inputs.shape)
            # sitk.WriteImage(sitk.GetImageFromArray(val_inputs.detach().cpu().numpy()[0,0,...]), "./out_res.nii")
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 2, model)
            print(val_outputs.shape)
            sitk.WriteImage(sitk.GetImageFromArray(val_outputs.detach().cpu().numpy()[0, 0, ...]), "./out_res.nii")
            sitk.WriteImage(sitk.GetImageFromArray(val_outputs.detach().cpu().numpy()[0, 1, ...]), "./out_res1.nii")
            # val_outputs = model_seg(val_inputs, val_feat[0], val_feat[1])
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            print(len(val_output_convert), val_output_convert[0].shape)

            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
            )
            # hd_dist = compute_hausdorff_distance(val_output_convert[0], val_labels_convert[0], include_background=True,  percentile=95)
            # print(f"Hausdorff Distance 95: {hd_dist.item()}")
        dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)
    std = np.std(dice_vals)
    print(mean_dice_val, std)
    writer.add_scalar('Validation Segmentation Loss', mean_dice_val, global_step)
    return mean_dice_val


max_iterations = args.max_iter
print('Maximum Iterations for training: {}'.format(str(args.max_iter)))
eval_num = args.eval_step
post_label = AsDiscrete(to_onehot=out_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
model.load_state_dict(torch.load("./out_cube_seg/best_metric_model.pth"))
model.eval()
epoch_iterator_val = tqdm(
    val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
)
dice_val = validation(epoch_iterator_val)
metric_values.append(dice_val)
if dice_val > dice_val_best:
    dice_val_best = dice_val
    global_step_best = global_step
    # torch.save(
    #     model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
    # )
    print(
        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
            dice_val_best, dice_val
        )
    )





