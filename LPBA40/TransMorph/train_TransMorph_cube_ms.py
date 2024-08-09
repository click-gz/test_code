
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import LPBA40Datasets
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph_cube_ms as TransMorph
import SimpleITK as sitk

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
    weights = [1, 0.1]  # loss weights
    save_dir = 'TransMorph_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    if not os.path.exists('/data1/gz/checkpoint/lpba40/transmorph_cube/'+save_dir):
        os.makedirs('/data1/gz/checkpoint/lpba40/transmorph_cube/'+save_dir)

    lr = 0.0001 # learning rate
    epoch_start = 0
    max_epoch = 50 #max traning epoch
    cont_training = False #if continue training

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()

    updated_lr = lr

    '''
    Initialize training
    '''
    train_dir = '/data1/gz/data/LPBA40/train'
    train_set = LPBA40Datasets(glob.glob(train_dir + '/*.nii.gz'))
    train_size = int(len(train_set) * 0.7)
    val_size = len(train_set) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_set, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    fixed = sitk.GetArrayFromImage(sitk.ReadImage("/data1/gz/data/LPBA40/fixed.nii.gz"))[np.newaxis, np.newaxis, ...]
    fixed =  (np.max(fixed) - fixed) / (np.max(fixed) - np.min(fixed))
    fixed = torch.from_numpy(fixed).cuda().float()

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    nccloss = losses.NCCLoss().cuda()
    gradloss= losses.Grad3d(penalty='l2').cuda()
    best_dsc = 10000
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        epoch_iterator = tqdm(
            train_loader, desc="Training (X / X Steps) (dice=X.X)", dynamic_ncols=True
        )
        siml = []
        for step, batch in enumerate(epoch_iterator):
            idx += 1
            model.train()
            # adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in batch]
            x = data[0].float()

            x_in = torch.cat((x,fixed), dim=1)
            output = model(x_in)
            loss = 0
            loss_vals = []
            nloss = nccloss(output[0], fixed)*weights[0]

            loss = nloss + gradloss(output[1])*weights[1]
            loss_all.update(loss.item(), 1)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del x_in
            del output
            epoch_iterator.set_description(
                'Iter {} of {} loss {:.4f}, Img Sim: {:.6f}'.format(
                    idx, len(train_loader), loss.item(), nloss.item()
                )
            )
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        val_loss = []
        val_iterator = tqdm(
            val_loader, desc="Val (X / X Steps) (dice=X.X)", dynamic_ncols=True
        )
        for step, batch in enumerate(val_iterator):
            # adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in batch]
            x = data[0].float()

            x_in = torch.cat((x, fixed), dim=1)
            output = model(x_in)
            nloss = nccloss(output[0], fixed) * weights[0]
            loss = nloss + gradloss(output[1]) * weights[1]
            val_loss.append(loss.item())
            val_iterator.set_description(
                'Iter {} of {} loss {:.4f}, Img Sim: {:.6f}'.format(
                    step, len(val_loader), loss.item(), nloss.item()
                )
            )
        if np.mean(val_loss) < best_dsc:
            best_dsc = np.mean(val_loss)
            save_checkpoint({
                'state_dict': model.state_dict()
            }, save_dir='/data1/gz/checkpoint/lpba40/transmorph_cube_ms/' , filename='{}.pth.tar'.format(np.mean(val_loss)))



def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    # model_lists = natsorted(glob.glob(save_dir + '*'))
    # while len(model_lists) > max_model_num:
    #     os.remove(model_lists[0])
    #     model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 1
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