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
from Model.model_cube_ori import U_Network, SpatialTransformer
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
    checks = glob.glob("/data1/gz/checkpoint/lpba40/cube_ori/*.pth")
    for check in checks:
        print(check)
        UNet.load_state_dict(torch.load(check))
        nccl = []
        ssiml = []
        gradl = []
        dicel=[]
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
            grad_loss = grad_loss_fn(flow_m2f)
            ssim_loss = ssim_loss_fn(m2f, fixed)
            nccl.append(-sim_loss.item())
            gradl.append(grad_loss.item())
            ssiml.append(1 - ssim_loss.item())

            d = compute_label_dice(fixed_seg, seg_out)
            dicel.append(d)

        print("mean(ncc): ", np.mean(nccl), "   std(ncc): ", np.std(nccl))
        print("mean(ssim): ", np.mean(ssiml), "   std(ssim): ", np.std(ssiml))
        print("mean(grad): ", np.mean(gradl), "   std(grad): ", np.std(gradl))
        print("mean(dice): ", np.mean(dicel), "   std(dice): ", np.std(dicel))
        if np.mean(nccl) > best_ncc:
            bestpth = check
            best_ncc = np.mean(nccl)
    print(bestpth, " best ncc: ", best_ncc)


if __name__ == "__main__":
    test()
