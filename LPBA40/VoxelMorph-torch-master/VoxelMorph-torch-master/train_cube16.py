# python imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
from Model.datagenerators import LPBA40Datasets
from Model.model_cube16 import U_Network, SpatialTransformer
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
    train_dir = '/data1/gz/data/LPBA40/train'
    train_set = LPBA40Datasets(glob.glob(train_dir + '/*.nii.gz'))
    train_size = int(len(train_set) * 0.7)
    val_size = len(train_set) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_set, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    fixed = sitk.GetArrayFromImage(sitk.ReadImage("/data1/gz/data/LPBA40/fixed.nii.gz"))[np.newaxis, np.newaxis, ...]
    fixed = (np.max(fixed) - fixed)/(np.max(fixed) - np.min(fixed))
    fixed = torch.from_numpy(fixed).cuda().float()
    # 创建配准网络（UNet）和STN
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    img_size = (160, 192, 160)
    UNet = U_Network(len(img_size), nf_enc, nf_dec).to(device)
    STN = SpatialTransformer(img_size).to(device)
    UNet.train()
    STN.train()
    # 模型参数个数
    print("UNet: ", count_parameters(UNet))
    print("STN: ", count_parameters(STN))

    # Set optimizer and losses
    opt = Adam(UNet.parameters(), lr=args.lr)
    sim_loss_fn = losses.NCCLoss() if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.Grad()
    be = 10000
    # Training loop.
    for i in range(1, 35):
        epoch_iterator = tqdm(
            train_loader, desc="Training (X / X Steps) (dice=X.X)", dynamic_ncols=True
        )
        siml=[]
        for step, batch in enumerate(epoch_iterator):
            UNet.train()
            data = [t.cuda() for t in batch]
            x = data[0].float()
            # print(x.shape, fixed.shape)
            # Run the data through the model to produce warp and flow field
            flow_m2f = UNet(x, fixed)
            m2f = STN(x, flow_m2f)
            # Calculate loss
            sim_loss = sim_loss_fn(m2f, fixed)
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
        val_loss = []
        val_iterator = tqdm(
            val_loader, desc="Val (X / X Steps) (dice=X.X)", dynamic_ncols=True
        )
        for step, batch in enumerate(val_iterator):
            data = [t.cuda() for t in batch]
            x = data[0].float()
            flow_m2f = UNet(x, fixed)
            m2f = STN(x, flow_m2f)
            sim_loss = sim_loss_fn(m2f, fixed)
            grad_loss = grad_loss_fn(flow_m2f)
            loss = sim_loss + grad_loss
            val_loss.append(loss.item())
            val_iterator.set_description(
                'Iter {} of {} loss {:.4f}, Img Sim: {:.6f}'.format(
                    step, len(val_loader), loss.item(), sim_loss.item()
                )
            )
        if np.mean(val_loss) < be:
            be = np.mean(val_loss)
            save_file_name = os.path.join("/data1/gz/checkpoint/lpba40/vxm_cube16/", '%.6f.pth' % np.mean(siml))
            torch.save(UNet.state_dict(), save_file_name)



if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()



