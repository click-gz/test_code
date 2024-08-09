import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.morphology import erosion, dilation

from C2FViT_model_cube import C2F_ViT_stage, AffineCOMTransform, Center_of_mass_initial_pairwise
from Functions import save_img, load_4D, min_max_norm

import glob
from data.ImageFolder import LPBA40InfDatasets
from torch.utils.data import DataLoader
from loss import NCCLoss, SSIM3D, DSC
import SimpleITK as sitk


def compute_label_dice(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = list(range(1, 182))
    dice_lst = []
    for cls in cls_lst:
        dice = DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)


def boundary_loss(mask1, mask2, kernel_size=3):
    # 将mask转换为numpy数组，以便使用skimage库
    mask1_np = mask1.detach().cpu().numpy()
    mask2_np = mask2.detach().cpu().numpy()

    # 计算mask的边界
    boundary1 = dilation(mask1_np) - erosion(mask1_np)
    boundary2 = dilation(mask2_np) - erosion(mask2_np)

    # 将边界转换回Tensor
    boundary1 = torch.from_numpy(boundary1).to(mask1.device).float()
    boundary2 = torch.from_numpy(boundary2).to(mask2.device).float()

    # 计算边界之间的损失，这里使用L1损失作为示例
    boundary_diff = F.l1_loss(boundary1, boundary2, reduction='none')

    # 可以选择对差异进行平均或求和作为最终的损失值
    # 这里我们选择平均
    boundary_loss_value = boundary_diff.mean()

    return boundary_loss_value


import SimpleITK as sitk
import numpy as np
from surface_distance import *

from thop import profile
from thop import clever_format
def hd95(f, m):
    hd95 = 0
    count = 0
    for i in range(1, 181):
        if ((f == i).sum() == 0) or ((m == i).sum() == 0):
            continue
        # print(i)
        hd95 += compute_robust_hausdorff(compute_surface_distances((f == i), (m == i), np.ones(3)), 95.)
        count += 1
    hd95 /= count
    print(hd95, count)
    return hd95


def iou_score(m, f):
    smooth = 1e-5
    iou = 0
    count = 0
    for i in range(1, 41):
        # if ((f == i).sum() == 0) or ((m == i).sum() == 0):
        #     continue
        intersection = ((f == i) & (m == i)).sum()
        union = ((f == i) | (m == i)).sum()
        iou += (intersection + smooth) / (union + smooth)
        count += 1
    print(iou / count, count)
    return iou / count


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--modelpath", type=str,
                        dest="modelpath",
                        default='../Model/C2FViT_affine_COM_pairwise_stagelvl3_118000.pth',
                        help="Pre-trained Model path")
    parser.add_argument("--savepath", type=str,
                        dest="savepath", default='/data1/gz/checkpoint/c2fvit_res',
                        help="path for saving images")
    parser.add_argument("--fixed", type=str,
                        dest="fixed", default='../Data/image_B.nii.gz',
                        help="fixed image")
    parser.add_argument("--moving", type=str,
                        dest="moving", default='../Data/image_A.nii.gz',
                        help="moving image")
    parser.add_argument("--com_initial", type=bool,
                        dest="com_initial", default=True,
                        help="True: Enable Center of Mass initialization, False: Disable")
    opt = parser.parse_args()

    savepath = opt.savepath
    fixed_path = opt.fixed
    moving_path = opt.moving
    com_initial = opt.com_initial
    # if not os.path.isdir(savepath):
    #     os.mkdir(savepath)

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    model = C2F_ViT_stage(img_size=(160, 160, 160), patch_size=[3, 7, 15], stride=[2, 4, 8], num_classes=12,
                          embed_dims=[128, 256, 256],
                          num_heads=[2, 2, 2], mlp_ratios=[2, 2, 2], qkv_bias=False, qk_scale=None, drop_rate=0.,
                          attn_drop_rate=0., norm_layer=nn.Identity,
                          depths=[4, 4, 4], sr_ratios=[1, 1, 1], num_stages=3, linear=False).to(device)
    input = torch.randn(2, 1, 1, 160, 160, 160).cuda()  # 随机生成一个输入张量，这个尺寸应该与模型输入的尺寸相匹配
    flops, params = profile(model, inputs=(input,))

    # 将结果转换为更易于阅读的格式
    flops, params = clever_format([flops, params], '%.3f')

    print(f"运算量：{flops}, 参数量：{params}")
    exit()
    print(f"Loading model weight {opt.modelpath} ...")

    model.eval()

    affine_transform = AffineCOMTransform(mode='nearest').cuda()
    init_center = Center_of_mass_initial_pairwise()

    test_dir = '/data1/gz/data/LPBA40/test'
    test_set = LPBA40InfDatasets(glob.glob(test_dir + '/*.nii.gz'))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    fixed = sitk.GetArrayFromImage(sitk.ReadImage("/data1/gz/data/LPBA40/fixed.nii.gz"))[np.newaxis, np.newaxis, ...]
    fixed = (fixed - np.min(fixed)) / (np.max(fixed) - np.min(fixed))
    fixed = torch.from_numpy(fixed).cuda().float()

    fixed_seg = \
        sitk.GetArrayFromImage(sitk.ReadImage("/data1/gz/data/LPBA40/label/S01.delineation.structure.label.nii.gz"))[
            np.newaxis, np.newaxis, ...]
    fixed_seg = torch.from_numpy(fixed_seg).float()

    sim_loss_fn = NCCLoss()
    ssim_loss_fn = SSIM3D()
    bestpth = ""
    best_ncc = 0
    with torch.no_grad():
        for checkpoint in glob.glob("/data1/gz/checkpoint/affine/lpba40/c2fvit_cube/*.pth"):
            print(checkpoint)
            model.load_state_dict(torch.load(checkpoint))
            # if com_initial:
            #     moving_img, init_flow = init_center(moving_img, fixed_img)
            nccl = []
            ssiml = []
            gradl = []
            dicel = []
            hd = []
            for X, seg in test_loader:
                x = X.cuda().float()
                seg = seg.cuda().float()
                X_d = F.interpolate(X, (160, 160, 160), mode="trilinear", align_corners=True).cuda().float()
                Y_d = F.interpolate(fixed, (160, 160, 160), mode="trilinear", align_corners=True).cuda().float()
                # seg = F.interpolate(seg, (160, 160, 160), mode="trilinear", align_corners=True).cuda().float()
                warpped_x_list, y_list, affine_para_list = model(X_d, Y_d)
                X_Y, affine_matrix = affine_transform(X_d, affine_para_list[-1])
                # sitk.WriteImage(sitk.GetImageFromArray(X_Y.detach().cpu().numpy()[0, 0, ...]),
                #                 "./out_cube.nii")
                x_seg_out, _ = affine_transform(seg, affine_para_list[-1])
                # sitk.WriteImage(sitk.GetImageFromArray(x_seg_out.detach().cpu().numpy()[0, 0, ...]), "./seg_out_cube.nii")

                # X_Y_cpu = X_Y.data.cpu().numpy()[0, 0, :, :, :]
                # loss_multiNCC = loss_similarity(warpped_x_list[-1], y_list[-1])
                # ll.append(loss_multiNCC.item())
                # save_img(X_Y_cpu, f"{savepath}/warped_{name.split('/')[-2]}.nii")
                sim_loss = sim_loss_fn(warpped_x_list[-1], y_list[-1])
                ssim_loss = ssim_loss_fn(warpped_x_list[-1], y_list[-1])
                nccl.append(-sim_loss.item())
                ssiml.append(1 - ssim_loss.item())

                d = compute_label_dice(fixed_seg, x_seg_out.detach().cpu())
                hdloss = hd95(fixed_seg.detach().cpu().numpy()[0, 0, ...], x_seg_out.detach().cpu().numpy()[0, 0, ...])
                dicel.append(d)
                hd.append(hdloss)
            print("mean(ncc): ", np.mean(nccl), "   std(ncc): ", np.std(nccl))
            print("mean(ssim): ", np.mean(ssiml), "   std(ssim): ", np.std(ssiml))
            # print("mean(grad): ", np.mean(gradl), "   std(grad): ", np.std(gradl))
            print("mean(dice): ", np.mean(dicel), "   std(boundary_loss): ", np.std(dicel))
            print("mean(dice): ", np.mean(hd), "   std(boundary_loss): ", np.std(hd))
            if np.mean(nccl) > best_ncc:
                bestpth = checkpoint
                best_ncc = np.mean(nccl)
        print(bestpth, " best ncc: ", best_ncc)