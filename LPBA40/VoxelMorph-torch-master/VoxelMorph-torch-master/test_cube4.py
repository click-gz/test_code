# python imports
import os
import glob
# external imports
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# internal imports
from Model import losses
from Model.config import args
from Model.model_cube4 import U_Network, SpatialTransformer
from Model.datagenerators import LPBA40InfDatasets
from torch.utils.data import DataLoader
def make_dirs():
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))


def compute_label_dice(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = list(range(1, 182))
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)

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

# @torchsnooper.snoop()
def test():
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(args.checkpoint_path)

    test_dir = '/data1/gz/data/LPBA40/test'
    test_set = LPBA40InfDatasets(glob.glob(test_dir + '/*.nii.gz'))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    fixed = sitk.GetArrayFromImage(sitk.ReadImage("/data1/gz/data/LPBA40/fixed.nii.gz"))[np.newaxis, np.newaxis, ...]
    fixed = (np.max(fixed) - fixed) / (np.max(fixed) - np.min(fixed))
    fixed = torch.from_numpy(fixed).cuda().float()

    fixed_seg = sitk.GetArrayFromImage(sitk.ReadImage("/data1/gz/data/LPBA40/label/S01.delineation.structure.label.nii.gz"))[np.newaxis, np.newaxis, ...]
    fixed_seg = torch.from_numpy(fixed_seg).float()

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    # Set up model
    img_size = (160, 192, 160)
    UNet = U_Network(len(img_size), nf_enc, nf_dec).to(device)
    # UNet.load_state_dict(torch.load(args.checkpoint_path))
    STN_img = SpatialTransformer(img_size).to(device)
    STN_label = SpatialTransformer(img_size, mode="bilinear")
    UNet.eval()
    STN_img.eval()
    # STN_label.eval()

    sim_loss_fn = losses.NCCLoss() if args.sim_loss == "ncc" else losses.mse_loss
    ssim_loss_fn = losses.SSIM3D()
    grad_loss_fn = losses.Grad()

    bestpth = ""
    best_ncc = 0
    checks = glob.glob("/data1/gz/checkpoint/lpba40/vxm_cube4/*.pth")
    for check in checks:
        print(check)
        UNet.load_state_dict(torch.load(check))
        nccl = []
        ssiml = []
        gradl = []
        dicel=[]
        hd = []
        epoch_iterator = tqdm(
            test_loader, desc="Training (X / X Steps) (dice=X.X)", dynamic_ncols=True
        )
        for step, batch in enumerate(epoch_iterator):
            UNet.train()
            data = [t.cuda() for t in batch]
            x = data[0].float()
            seg = data[1].detach().cpu().float()
            # print(x.shape)
            # print(x.shape, fixed.shape)
            # Run the data through the model to produce warp and flow field
            flow_m2f = UNet(x, fixed)
            m2f = STN_img(x, flow_m2f)
            seg_out = STN_label(seg, flow_m2f.detach().cpu())
            # Calculate loss
            sim_loss = sim_loss_fn(m2f, fixed)
            # grad_loss = jacobian_determinant(flow_m2f.detach().cpu())
            jac_det = (jacobian_determinant(flow_m2f.detach().cpu()) + 1)
            grad_loss = np.log(jac_det)
            ssim_loss = ssim_loss_fn(m2f, fixed)
            nccl.append(-sim_loss.item())
            gradl.append(grad_loss)
            ssiml.append(1 - ssim_loss.item())

            d = compute_label_dice(fixed_seg, seg_out)
            hdloss = hd95(fixed_seg.numpy()[0,0,...], seg_out.numpy()[0,0,...])
            dicel.append(d)
            hd.append(hdloss)

        print("mean(ncc): ", np.mean(nccl), "   std(ncc): ", np.std(nccl))
        print("mean(ssim): ", np.mean(ssiml), "   std(ssim): ", np.std(ssiml))
        print("mean(grad): ", np.mean(gradl), "   std(grad): ", np.std(gradl))
        print("mean(dice): ", np.mean(dicel), "   std(dice): ", np.std(dicel))
        print("mean(hd): ", np.mean(hd), "   std(hd): ", np.std(hd))
        if np.mean(nccl) > best_ncc:
            bestpth = check
            best_ncc = np.mean(nccl)
    print(bestpth, " best ncc: ", best_ncc)


if __name__ == "__main__":
    test()
