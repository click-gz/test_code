import glob
import os, losses, utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import LPBA40InfDatasets
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from models import PAN
import nibabel
import SimpleITK as sitk
import time
from Evaluation.Eval_metrics import compute_surface_distances, \
    compute_average_surface_distance, compute_robust_hausdorff, compute_dice_coefficient
from thop import profile
from thop import clever_format
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def compute_label_dice(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = list(range(1, 182))
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)



def make_one_hot(mask, num_class):
    # 数据转为one hot 类型
    # mask_unique = np.unique(mask)
    mask_unique = [m for m in range(num_class)]
    one_hot_mask = [mask == i for i in mask_unique]
    one_hot_mask = np.stack(one_hot_mask)
    return one_hot_mask


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
    num_class = 57
    test_dir = '/LPBA_path/Test/'
    imgs_path = glob.glob(os.path.join(test_dir, "*.nii.gz"))
    imgs_path.sort()
    img_size = (160, 192, 160)
    spacing = (1, 1, 1)

    model_idx = -1
    model_root = '/LPBA_path/Model/PAN'
    model_path = model_root + '/experiments/'
    # log_path = save_root + '/logs' + save_dir


    model_folder = 'PAN/'
    model_dir = model_path + model_folder


    model = PAN(img_size)
    model.cuda()
    input = torch.randn(2, 1, 1, 160, 192, 160).cuda()  # 随机生成一个输入张量，这个尺寸应该与模型输入的尺寸相匹配
    flops, params = profile(model, inputs=(input,))

    # 将结果转换为更易于阅读的格式
    flops, params = clever_format([flops, params], '%.3f')

    print(f"运算量：{flops}, 参数量：{params}")
    exit()
    reg_model = utils.register_model(img_size, 'bilinear')
    reg_model.cuda()
    reg_model_m = utils.register_model(img_size, 'bilinear')
    reg_model_m.cuda()

    test_dir = '/data1/gz/data/LPBA40/test'
    test_set = LPBA40InfDatasets(glob.glob(test_dir + '/*07*.nii.gz'))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    fixed = sitk.GetArrayFromImage(sitk.ReadImage("/data1/gz/data/LPBA40/fixed.nii.gz"))[np.newaxis, np.newaxis, ...]
    fixed = (np.max(fixed) - fixed) / (np.max(fixed) - np.min(fixed))
    fixed = torch.from_numpy(fixed).cuda().float()

    fixed_seg = \
        sitk.GetArrayFromImage(sitk.ReadImage("/data1/gz/data/LPBA40/label/S01.delineation.structure.label.nii.gz"))[
            np.newaxis, np.newaxis, ...]
    fixed_seg = torch.from_numpy(fixed_seg).float()


    sim_loss_fn = losses.NCCLoss()
    ssim_loss_fn = losses.SSIM3D()
    grad_loss_fn = losses.Grad()

    bestpth = ""
    best_ncc = 0
    checks = glob.glob("/data1/gz/checkpoint/lpba40/pan/*.pth.tar")
    with torch.no_grad():
        for check in checks:
            print(check)
            model.load_state_dict(torch.load(check)['state_dict'])
            nccl = []
            ssiml = []
            gradl = []
            dicel = []
            hd = []
            epoch_iterator = tqdm(
                test_loader, desc="Testing (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            for step, batch in enumerate(epoch_iterator):
                model.eval()
                torch.cuda.empty_cache()
                data = [t.cuda() for t in batch]
                x = data[0].float()
                seg = data[1].cuda().float()

                # x_in = torch.cat((x,y),dim=1)
                x_def, flow, _, _ = model(x, fixed)
                sitk.WriteImage(sitk.GetImageFromArray(x_def.detach().cpu().numpy()[0, 0, ...]), "./out.nii")
                def_out = reg_model([seg.cuda().float(), flow.cuda()])
                flow = flow.permute(0, 2, 3, 4, 1).cpu().detach().numpy()[0]
                i = sitk.GetImageFromArray(flow, isVector=True)
                sitk.WriteImage(i, "./disp.nii")
                sitk.WriteImage(sitk.GetImageFromArray(def_out.detach().cpu().numpy()[0, 0, ...]), "./seg_out.nii")


                sim_loss = sim_loss_fn(x_def, fixed)
                jac_det = (jacobian_determinant(flow.detach().cpu()) + 1).clip(0.000000001, 1000000000)
                # print(jac_det)
                grad_loss = np.log(jac_det)
                ssim_loss = ssim_loss_fn(x_def, fixed)
                nccl.append(-sim_loss.item())
                gradl.append(grad_loss)
                ssiml.append(1 - ssim_loss.item())

                d = compute_label_dice(fixed_seg, def_out.detach().cpu().float())
                hdloss = hd95(fixed_seg.detach().cpu().numpy()[0, 0, ...], def_out.detach().cpu().numpy()[0, 0, ...])
                dicel.append(d)
                hd.append(hdloss)

            print("mean(ncc): ", np.mean(nccl), "   std(ncc): ", np.std(nccl))
            print("mean(ssim): ", np.mean(ssiml), "   std(ssim): ", np.std(ssiml))
            print("mean(grad): ", np.mean(gradl), "   std(grad): ", np.std(gradl))
            print("mean(dice): ", np.mean(dicel), "   std(dice): ", np.std(dicel))
            print("mean(dice): ", np.mean(hd), "   std(dice): ", np.std(hd))
            if np.mean(nccl) > best_ncc:
                bestpth = check
                best_ncc = np.mean(nccl)
    print(bestpth, " best ncc: ", best_ncc)



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