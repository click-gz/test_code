import glob
import os, losses, utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import LPBA40InfOriDatasets
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph_cube1 as TransMorph
import SimpleITK as sitk

def compute_label_dice(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = list(range(1, 182))
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)


def main():
    test_dir = 'D:/DATA/JHUBrain/Test/'
    model_idx = -1
    weights = [1, 0.02]




    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    model.cuda()
    reg_model = utils.register_model((160, 192, 160), 'bilinear')
    def_model = utils.register_model((160, 192, 160))
    # reg_model.cuda()

    test_dir = '/data1/gz/data/LPBA40/test'
    test_set = LPBA40InfOriDatasets(glob.glob(test_dir + '/*.nii.gz'))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    fixed = sitk.GetArrayFromImage(sitk.ReadImage("/data1/gz/data/LPBA40/fixed.nii.gz"))[np.newaxis, np.newaxis, ...]
    nfixed = (np.max(fixed) - fixed) / (np.max(fixed) - np.min(fixed))
    nfixed, fixed = torch.from_numpy(nfixed).cuda().float(), torch.from_numpy(fixed).cuda().float()

    fixed_seg = \
    sitk.GetArrayFromImage(sitk.ReadImage("/data1/gz/data/LPBA40/label/S01.delineation.structure.label.nii.gz"))[
        np.newaxis, np.newaxis, ...]
    fixed_seg = torch.from_numpy(fixed_seg).float()

    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    sim_loss_fn = losses.NCCLoss()
    ssim_loss_fn = losses.SSIM3D()
    grad_loss_fn = losses.Grad()
    bestpth = ""
    best_ncc = 0
    checks = glob.glob("/data1/gz/checkpoint/lpba40/transmorph_cube1/*/*.pth.tar")
    for check in checks:
        print(check)
        model.load_state_dict(torch.load(check)['state_dict'])
        nccl = []
        ssiml = []
        gradl = []
        dicel = []
        with torch.no_grad():
            idx = 0
            epoch_iterator = tqdm(
                test_loader, desc="Training (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            siml = []
            for step, batch in enumerate(epoch_iterator):
                idx += 1
                model.eval()
                data = [t.cuda() for t in batch]
                x = data[0].float()
                seg = data[1].cuda().float()
                orim = data[2].float()

                x_in = torch.cat((x, nfixed), dim=1)
                output = model(x_in)
                seg_out = reg_model(seg, output[1])

                res = def_model(orim, output[1])

                sim_loss = sim_loss_fn(res, fixed)
                grad_loss = grad_loss_fn(output[1])
                ssim_loss = ssim_loss_fn(res, fixed)
                nccl.append(-sim_loss.item())
                gradl.append(grad_loss.item())
                ssiml.append(1 - ssim_loss.item())

                d = compute_label_dice(fixed_seg, seg_out.detach().cpu().float())
                dicel.append(d)

            print("mean(ncc): ", np.mean(nccl), "   std(ncc): ", np.std(nccl))
            print("mean(ssim): ", np.mean(ssiml), "   std(ssim): ", np.std(ssiml))
            print("mean(grad): ", np.mean(gradl), "   std(grad): ", np.std(gradl))
            print("mean(dice): ", np.mean(dicel), "   std(dice): ", np.std(dicel))
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