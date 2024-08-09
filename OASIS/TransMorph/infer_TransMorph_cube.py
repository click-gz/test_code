import glob
import os, losses, utils

from skimage.morphology import dilation, erosion
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import OASISBrainInferDataset
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph_cube as TransMorph
import torch.nn.functional as F

def compute_label_dice(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = list(range(1, 2))
    print(cls_lst)
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
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
    iou=0
    count=0
    for i in range(1, 41):
        # if ((f == i).sum() == 0) or ((m == i).sum() == 0):
        #     continue
        intersection = ((f==i) & (m==i)).sum()
        union = ((f==i) |  (m==i)).sum()
        iou += (intersection + smooth) / (union + smooth)
        count += 1
    print(iou/count, count)
    return iou/count

import scipy.ndimage
def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape

    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)

    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (
                jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) - \
             jacobian[1, 0, :, :, :] * (
                         jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :,
                                                                                                       :, :]) + \
             jacobian[2, 0, :, :, :] * (
                         jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :,
                                                                                                       :, :])

    return jacdet

def main():
    test_dir = 'D:/DATA/JHUBrain/Test/'
    model_idx = -1
    weights = [1, 0.02]

    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    model.cuda()
    reg_model = utils.register_model((192, 224, 160), 'nearest')
    reg_model.cuda()

    test_dir = '/data1/gz/data/OASIS/imagesTs'
    test_set = OASISBrainInferDataset(glob.glob(test_dir + '/*045*.nii.gz'))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    sim_loss_fn = losses.NCCLoss()
    ssim_loss_fn = losses.SSIM3D()
    grad_loss_fn = losses.Grad()

    bestpth = ""
    best_ncc = 0
    checks = glob.glob("/data1/gz/checkpoint/oasis/transmorph_cube/*/*.pth.tar")
    with torch.no_grad():
        for check in checks:
            print(check)
            model.load_state_dict(torch.load(check)['state_dict'])
            nccl = []
            ssiml = []
            gradl = []
            bl = []
            hd=[]
            epoch_iterator = tqdm(
                test_loader, desc="Testing (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            for step, batch in enumerate(epoch_iterator):
                data = [t.cuda() for t in batch]
                x = data[0].float()
                y = data[1].float()
                x_seg = data[2].float()
                y_seg = data[3].float()

                x_in = torch.cat((x,y),dim=1)
                x_def, flow = model(x_in)
                def_out = reg_model(x_seg, flow)
                sitk.WriteImage(sitk.GetImageFromArray(x_def.detach().cpu().numpy()[0, 0, ...]), "./out_cube.nii")
                flow = flow.permute(0, 2, 3, 4, 1).cpu().detach().numpy()[0]
                i = sitk.GetImageFromArray(flow, isVector=True)
                sitk.WriteImage(i, "./disp_cube.nii")
                break
                sim_loss = sim_loss_fn(x_def, y)
                jac_det = (jacobian_determinant(flow.detach().cpu()) + 1).clip(0.000000001, 1000000000)
                # print(jac_det)
                grad_loss = np.log(jac_det)
                ssim_loss = ssim_loss_fn(x_def, y)
                nccl.append(-sim_loss.item())
                gradl.append(grad_loss)
                ssiml.append(1 - ssim_loss.item())

                # b = boundary_loss(x_seg, y_seg)
                # bl.append(b.detach().cpu())
                d = compute_label_dice(y_seg.detach().cpu(), def_out.detach().cpu().float())
                hdloss = hd95(y_seg.detach().cpu().numpy()[0, 0, ...], def_out.detach().cpu().numpy()[0, 0, ...])
                bl.append(d)
                hd.append(hdloss)
            print("mean(ncc): ", np.mean(nccl), "   std(ncc): ", np.std(nccl))
            print("mean(ssim): ", np.mean(ssiml), "   std(ssim): ", np.std(ssiml))
            print("mean(grad): ", np.mean(gradl), "   std(grad): ", np.std(gradl))
            print("mean(boundary_loss): ", np.mean(bl), "   std(boundary_loss): ", np.std(bl))
            print("mean(boundary_loss): ", np.mean(hd), "   std(boundary_loss): ", np.std(hd))
            if np.mean(nccl) > best_ncc:
                bestpth = check
                best_ncc = np.mean(nccl)
        print(bestpth, " best ncc: ", best_ncc)

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()