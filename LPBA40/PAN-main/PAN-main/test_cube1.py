import glob
import os, losses, utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import LPBA40InfDatasets
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from models_cube1 import PAN
import nibabel
import SimpleITK as sitk
import time
from Evaluation.Eval_metrics import compute_surface_distances, \
    compute_average_surface_distance, compute_robust_hausdorff, compute_dice_coefficient

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
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    reg_model_m = utils.register_model(img_size, 'bilinear')
    reg_model_m.cuda()

    test_dir = '/data1/gz/data/LPBA40/test'
    test_set = LPBA40InfDatasets(glob.glob(test_dir + '/*.nii.gz'))
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
    checks = glob.glob("/data1/gz/checkpoint/lpba40/pan_cube1/*.pth.tar")
    with torch.no_grad():
        for check in checks:
            print(check)
            model.load_state_dict(torch.load(check)['state_dict'])
            nccl = []
            ssiml = []
            gradl = []
            dicel = []
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
                x_def, flow= model(x, fixed)
                def_out = reg_model([seg.cuda().float(), flow.cuda()])
                out = reg_model_m([x.cuda().float(), flow.cuda()])

                sim_loss = sim_loss_fn(x_def, fixed)
                grad_loss = grad_loss_fn(flow)
                ssim_loss = ssim_loss_fn(x_def, fixed)
                nccl.append(-sim_loss.item())
                gradl.append(grad_loss.item())
                ssiml.append(1 - ssim_loss.item())

                d = compute_label_dice(fixed_seg, def_out.detach().cpu().float())
                dicel.append(d)

            print("mean(ncc): ", np.mean(nccl), "   std(ncc): ", np.std(nccl))
            print("mean(ssim): ", np.mean(ssiml), "   std(ssim): ", np.std(ssiml))
            print("mean(grad): ", np.mean(gradl), "   std(grad): ", np.std(gradl))
            print("mean(dice): ", np.mean(dicel), "   std(dice): ", np.std(dicel))
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