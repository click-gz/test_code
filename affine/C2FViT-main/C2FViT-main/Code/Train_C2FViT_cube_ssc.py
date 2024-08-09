import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from C2FViT_model_cube import C2F_ViT_stage, AffineCOMTransform, Center_of_mass_initial_pairwise, multi_resolution_NCC
from Functions import Dataset_epoch
from data.ImageFolder import OASISBrainDataset, OASISBrainInferDataset
from loss import NCCLoss

def dice(im1, atlas):
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if (i == 0) or ((im1 == i).sum() == 0) or ((atlas == i).sum() == 0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
    return dice / num_count


def train():
    print("Training C2FViT...")
    model = C2F_ViT_stage(img_size=(160, 160, 160), patch_size=[3, 7, 15], stride=[2, 4, 8], num_classes=12,
                          embed_dims=[128, 256, 256],
                          num_heads=[2, 2, 2], mlp_ratios=[2, 2, 2], qkv_bias=False, qk_scale=None, drop_rate=0.,
                          attn_drop_rate=0., norm_layer=nn.Identity,
                          depths=[4, 4, 4], sr_ratios=[1, 1, 1], num_stages=3, linear=False).cuda()

    # model = C2F_ViT_stage(img_size=128, patch_size=[7, 15], stride=[4, 8], num_classes=12, embed_dims=[256, 256],
    #                       num_heads=[2, 2], mlp_ratios=[2, 2], qkv_bias=False, qk_scale=None, drop_rate=0.,
    #                       attn_drop_rate=0., norm_layer=nn.Identity, depths=[4, 4], sr_ratios=[1, 1], num_stages=2,
    #                       linear=False).cuda()

    # model = C2F_ViT_stage(img_size=128, patch_size=[15], stride=[8], num_classes=12, embed_dims=[256],
    #                       num_heads=[2], mlp_ratios=[2], qkv_bias=False, qk_scale=None, drop_rate=0.,
    #                       attn_drop_rate=0., norm_layer=nn.Identity, depths=[4], sr_ratios=[1], num_stages=1,
    #                       linear=False).cuda()

    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    affine_transform = AffineCOMTransform().cuda()
    init_center = Center_of_mass_initial_pairwise()

    loss_similarity = NCCLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_dir = '/data1/gz/data/OASIS/imagesTr'
    train_set =  OASISBrainDataset(glob.glob(train_dir + '/*.nii.gz'))
    train_size = int(len(train_set) * 0.7)
    val_size = len(train_set) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_set, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(0))

    train_loader = Data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    test_dir = '/data1/gz/data/OASIS/imagesTs'
    test_set = OASISBrainInferDataset(glob.glob(test_dir + '/*.nii.gz'))
    val_loader = Data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    model_dir = "/data1/gz/checkpoint/affine/oasis/c2fvit_cube_ssc"
    # model.load_state_dict(torch.load("/data1/gz/checkpoint/affine/oasis/c2fvit/C2FViT_affine_COM_pairwise_stagelvl3_65500.pth"))
    step = 0
    be = 1000
    lossall = np.zeros((2, iteration + 1))
    while step <= iteration:
        train_loss = []
        for X, Y in train_loader:
            X = X.cuda().float()
            Y = Y.cuda().float()

            X = F.interpolate(X, (160, 160, 160), mode="trilinear", align_corners=True)
            Y = F.interpolate(Y, (160, 160, 160), mode="trilinear", align_corners=True)

            warpped_x_list, y_list, affine_para_list = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(warpped_x_list[-1], y_list[-1])

            loss = loss_multiNCC
            train_loss.append(loss.item())
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}"'.format(
                    step, loss.item(), loss_multiNCC.item()))
            sys.stdout.flush()
            step += 1
        val_loss = []
        with torch.no_grad():
            for X, Y, _, _ in val_loader:
                X = X.cuda().float()
                Y = Y.cuda().float()
                X = F.interpolate(X, (160, 160, 160), mode="trilinear", align_corners=True)
                Y = F.interpolate(Y, (160, 160, 160), mode="trilinear", align_corners=True)
                warpped_x_list, y_list, affine_para_list = model(X, Y)
                loss_multiNCC = loss_similarity(warpped_x_list[-1], y_list[-1])
                val_loss.append(loss_multiNCC.item())
            if np.mean(val_loss) < be:
                be = np.mean(val_loss)
                modelname = model_dir + '/' + model_name + "stagelvl3ssc_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
        if step > iteration:
            break
        print("one epoch pass: ", np.mean(train_loss), np.mean(val_loss), be)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--modelname", type=str,
                        dest="modelname",
                        default='C2FViT_affine_COM_pairwise_',
                        help="Model name")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=0.0001, help="learning rate")
    parser.add_argument("--iteration", type=int,
                        dest="iteration", default=10000,
                        help="number of total iterations")
    parser.add_argument("--checkpoint", type=int,
                        dest="checkpoint", default=414,
                        help="frequency of saving models")
    parser.add_argument("--datapath", type=str,
                        dest="datapath",
                        default='/PATH/TO/YOUR/DATA',
                        help="data path for training images")
    parser.add_argument("--com_initial", type=bool,
                        dest="com_initial", default=True,
                        help="True: Enable Center of Mass initialization, False: Disable")
    opt = parser.parse_args()

    lr = opt.lr
    iteration = opt.iteration
    n_checkpoint = opt.checkpoint
    datapath = opt.datapath
    com_initial = opt.com_initial

    model_name = opt.modelname


    print("Training %s ..." % model_name)
    train()


