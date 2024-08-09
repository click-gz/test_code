import glob
import os, losses, utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import LPBA40InfDatasets
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
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
    reg_model = utils.register_model((160, 192, 160), 'nearest')
    # reg_model.cuda()

    moving = sitk.GetArrayFromImage(sitk.ReadImage("./moving.nii.gz"))[np.newaxis, np.newaxis, ...]
    moving = (np.max(moving) - moving) / (np.max(moving) - np.min(moving))
    moving = torch.from_numpy(moving).cuda().float()

    fixed = sitk.GetArrayFromImage(sitk.ReadImage("./fixed.nii.gz"))[np.newaxis, np.newaxis, ...]
    fixed = (np.max(fixed) - fixed) / (np.max(fixed) - np.min(fixed))
    fixed = torch.from_numpy(fixed).cuda().float()


    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    sim_loss_fn = losses.NCCLoss()
    ssim_loss_fn = losses.SSIM3D()
    grad_loss_fn = losses.Grad()
    bestpth = ""
    best_ncc = 0
    model.load_state_dict(torch.load("./dsc-0.993.pth.tar")['state_dict'])
    x_in = torch.cat((moving, fixed), dim=1)
    output = model(x_in)

    flow = output[1].permute(0, 2,3,4,1).cpu().detach().numpy()[0]
    # flow = np.array([zoom(flow[i], 0.5, order=2) for i in range(3)]).astype(np.float16)
    print(flow.shape)
    i = sitk.GetImageFromArray(flow, isVector=True)
    sitk.WriteImage(i, "./disp.nii.gz")

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