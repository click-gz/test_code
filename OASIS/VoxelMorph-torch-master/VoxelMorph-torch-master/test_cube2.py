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
from Model.model_cube2 import U_Network, SpatialTransformer
from Model.datagenerators import OASISBrainInferDataset
from torch.utils.data import DataLoader

import torch.nn.functional as F
from skimage.morphology import dilation, erosion

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
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
               63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
               163, 164, 165, 166]
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

# @torchsnooper.snoop()
def test():
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(args.checkpoint_path)

    test_dir = '/data1/gz/data/OASIS/imagesTs'
    test_set = OASISBrainInferDataset(glob.glob(test_dir + '/*.nii.gz'))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    # Set up model
    img_size = (192, 224, 160)
    UNet = U_Network(len(img_size), nf_enc, nf_dec).to(device)
    # UNet.load_state_dict(torch.load(args.checkpoint_path))
    STN_img = SpatialTransformer(img_size).to(device)
    STN_label = SpatialTransformer(img_size, mode="nearest").to(device)
    UNet.eval()
    STN_img.eval()
    STN_label.eval()

    sim_loss_fn = losses.NCCLoss() if args.sim_loss == "ncc" else losses.mse_loss
    ssim_loss_fn = losses.SSIM3D()
    grad_loss_fn = losses.Grad()

    bestpth = ""
    best_ncc=0
    checks = glob.glob("/data1/gz/checkpoint/oasis/vxm_cube2/*.pth")
    with torch.no_grad():
        for check in checks:
            UNet.load_state_dict(torch.load(check))
            nccl = []
            ssiml=[]
            gradl=[]
            bl = []
            epoch_iterator = tqdm(
                test_loader, desc="Training (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            for step, batch in enumerate(epoch_iterator):
                data = [t.cuda() for t in batch]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]

                # 获得配准后的图像和label
                pred_flow = UNet(x, y)
                pred_img = STN_img(x, pred_flow)
                # pred_label = STN_label(x_seg, pred_flow)

                sim_loss = sim_loss_fn(pred_img, y)
                grad_loss = grad_loss_fn(pred_flow)
                ssim_loss = ssim_loss_fn(pred_img, y)
                nccl.append(-sim_loss.item())
                gradl.append(grad_loss.item())
                ssiml.append(1-ssim_loss.item())

                b  = boundary_loss(x_seg, y_seg)
                bl.append(b.detach().cpu())
                # if '7' in file:
                #     save_image(pred_img, f_img, "7_warped.nii.gz")
                #     save_image(pred_flow.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img, "7_flow.nii.gz")
                #     save_image(pred_label, f_img, "7_label.nii.gz")
                del pred_flow, pred_img

            print("mean(ncc): ", np.mean(nccl), "   std(ncc): ", np.std(nccl))
            print("mean(ssim): ", np.mean(ssiml), "   std(ssim): ", np.std(ssiml))
            print("mean(grad): ", np.mean(gradl), "   std(grad): ", np.std(gradl))
            print("mean(boundary_loss): ", np.mean(bl), "   std(boundary_loss): ", np.std(bl))
            if np.mean(nccl) > best_ncc:
                bestpth =check
                best_ncc = np.mean(nccl)
        print(bestpth, " best ncc: ", best_ncc)


if __name__ == "__main__":
    test()
