# python imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import glob
import warnings
# external imports
import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
from tqdm import tqdm

# internal imports
from Model import losses
from Model.config import args
from Model.datagenerators import OASISBrainDataset, OASISBrainInferDataset
from Model.model_cube import U_Network, SpatialTransformer
from torch.utils.data import DataLoader

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))


def train():
    # 创建需要的文件夹并指定gpu
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    train_dir = '/data1/gz/data/OASIS/imagesTr'
    train_set = OASISBrainDataset(glob.glob(train_dir + '/*.nii.gz'))
    train_size = int(len(train_set) * 0.7)
    val_size = len(train_set) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_set, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(0))
    train_loader = Data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    test_dir = '/data1/gz/data/OASIS/imagesTs'
    test_set = OASISBrainInferDataset(glob.glob(test_dir + '/*.nii.gz'))
    val_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    # val_loader = Data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # 创建配准网络（UNet）和STN
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    img_size = (192, 224, 160)
    UNet = U_Network(len(img_size), nf_enc, nf_dec).to(device)
    STN = SpatialTransformer(img_size).to(device)
    UNet.train()
    STN.train()
    # 模型参数个数
    print("UNet: ", count_parameters(UNet))
    print("STN: ", count_parameters(STN))

    # Set optimizer and losses
    opt = Adam(UNet.parameters(), lr=1e-4)
    sim_loss_fn = losses.NCCLoss() if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.Grad()
    be = 10000
    # Training loop.
    i = 0
    lr=0.0001
    while i < args.n_iter:
        epoch_iterator = tqdm(
            train_loader, desc="Training (X / X Steps) (dice=X.X)", dynamic_ncols=True
        )
        siml = []
        for step, batch in enumerate(epoch_iterator):
            i += 1
            UNet.train()
            data = [t.cuda() for t in batch]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            # Run the data through the model to produce warp and flow field
            flow_m2f = UNet(x, y)
            m2f = STN(x, flow_m2f)
            # Calculate loss
            sim_loss = sim_loss_fn(m2f, y)
            grad_loss = grad_loss_fn(flow_m2f)
            loss = sim_loss + grad_loss
            siml.append(sim_loss.item())
            epoch_iterator.set_description(
                "i: %d  loss: %f  sim: %f  grad: %f" % (i, loss.item(), sim_loss.item(), grad_loss.item())
            )
            # print("%d, %f, %f, %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), file=f)
            # Backwards and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_iterator = tqdm(
            val_loader, desc="Training (X / X Steps) (dice=X.X)", dynamic_ncols=True
        )
        val_loss = []
        with torch.no_grad():
            for step, batch in enumerate(val_iterator):
                data = [t.cuda() for t in batch]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                # Run the data through the model to produce warp and flow field
                flow_m2f = UNet(x, y)
                m2f = STN(x, flow_m2f)
                # Calculate loss
                sim_loss = sim_loss_fn(m2f, y)
                grad_loss = grad_loss_fn(flow_m2f)
                loss = sim_loss + grad_loss
                val_loss.append(loss.item())
            if np.mean(val_loss) < be:
                be = np.mean(val_loss)
                save_file_name = os.path.join(args.model_dir, '%.6f.pth' % np.mean(val_loss))
                torch.save(UNet.state_dict(), save_file_name)

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
