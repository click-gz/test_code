import glob
import os, losses, utils
import sys
from torch.utils.data import DataLoader
from data.datasets import LPBA40Datasets
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models_cube2 import PAN
import random
import SimpleITK as sitk

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    batch_size = 1
    num_class = 57

    weights = [1, 1, 1, 1]  # loss weights
    lr = 0.0001
    save_dir = '/lpba40/pan_cube2/'
    save_root = '/data1/gz/checkpoint/'
    save_exp = save_root  + save_dir
    if not os.path.exists(save_exp):
        os.makedirs(save_exp)
    lr = 0.0001
    epoch_start = 0
    max_epoch = 500
    img_size =(160, 192, 160)
    cont_training = False

    '''
    Initialize model
    '''
    model = PAN(img_size)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    # reg_model_bilin = utils.register_model(img_size, 'bilinear')
    # reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 0
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        best_model = torch.load(save_exp + natsorted(os.listdir(save_exp))[-3])['state_dict']
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''

    train_dir = '/data1/gz/data/LPBA40/train'
    train_set = LPBA40Datasets(glob.glob(train_dir + '/*.nii.gz'))
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    fixed = sitk.GetArrayFromImage(sitk.ReadImage("/data1/gz/data/LPBA40/fixed.nii.gz"))[np.newaxis, np.newaxis, ...]
    fixed = (np.max(fixed) - fixed) / (np.max(fixed) - np.min(fixed))
    fixed = torch.from_numpy(fixed).cuda().float()

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    # criterion = nn.MSELoss()
    nccloss = losses.NCCLoss().cuda()
    gradloss = losses.Grad3d(penalty='l2').cuda()
    best_dsc = 10000
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0].float()
            # x_in = torch.cat((x, y), dim=1)
            output = model(x, fixed)
            loss = 0
            loss_vals = []
            nloss = nccloss(output[0], fixed) * weights[0]
            loss = nloss + gradloss(output[1]) * weights[1]
            loss_all.update(loss.item(), 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}'.format(idx, len(train_loader), loss.item(),
                                                                                   nloss.item()))

        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        if loss_all.avg < best_dsc:
            best_dsc = loss_all.avg
            save_checkpoint({
                'state_dict': model.state_dict()
            }, save_dir='/data1/gz/checkpoint'+save_dir, filename='dsc{:.3f}.pth.tar'.format(loss_all.avg))

        # plt.switch_backend('agg')
        # pred_fig = comput_fig(def_out)
        # grid_fig = comput_fig(def_grid)
        # x_fig = comput_fig(x_seg)
        # tar_fig = comput_fig(y_seg)
        # writer.add_figure('Grid', grid_fig, epoch)
        # plt.close(grid_fig)
        # writer.add_figure('input', x_fig, epoch)
        # plt.close(x_fig)
        # writer.add_figure('ground truth', tar_fig, epoch)
        # plt.close(tar_fig)
        # writer.add_figure('prediction', pred_fig, epoch)
        # plt.close(pred_fig)
        loss_all.reset()
def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=5):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))


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