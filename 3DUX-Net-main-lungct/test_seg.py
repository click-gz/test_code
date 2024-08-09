from monai.utils import first, set_determinism

from monai.transforms import AsDiscrete
from networks.UXNet_3D.network_backbone import UXNET
# from networks.UXNet_3D.network_cube_backbone import UXNET
from monai.networks.nets import SwinUNETR
# from networks.SwinUNETR.swunetr1 import SwinUNETR
from networks.UNETR.unetr_cube import UNETR
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.nnFormer.nnFormer_Wcube_seg import nnFormer
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch

import torch
from load_datasets_transforms import data_loader, data_transforms, infer_post_transforms

import os
import argparse

parser = argparse.ArgumentParser(description='3D UX-Net inference hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='/data1/gz/data/Lung250M', required=False, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='/data1/gz/data/out1_seg', required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='lung', required=False, help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='UNETR', required=False, help='Network models: '
                                                                                     '{TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET}')
parser.add_argument('--trained_weights', default='/data1/gz/data/out1/best_metric_model.pth', required=False, help='Path of pretrained/fine-tuned weights')
parser.add_argument('--mode', type=str, default='test_seg', help='Training or testing mode')
parser.add_argument('--sw_batch_size', type=int, default=4, help='Sliding window batch size for inference')
parser.add_argument('--overlap', type=float, default=0.5, help='Sub-volume overlapped percentage')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='1', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.01, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

_, test_samples, out_classes = data_loader(args)

test_files = [
    {"image": image_name} for image_name in zip(test_samples['images'])
]

set_determinism(seed=0)

test_transforms = data_transforms(args)
post_transforms = infer_post_transforms(args, test_transforms, out_classes)

## Inference Pytorch Data Loader and Caching
test_ds = CacheDataset(
    data=test_files, transform=test_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)

test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers)

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

model.load_state_dict(torch.load("./uneout_seg_cube/best_metric_model.pth"))
model.eval()
with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        images = test_data["image"].to(device)
        roi_size = (96, 96, 96)
        test_data['pred'] = sliding_window_inference(
            images, roi_size, args.sw_batch_size, model, overlap=args.overlap
        )

        test_data = [post_transforms(i) for i in decollate_batch(test_data)]
        # print("res: ", test_data.shape)
