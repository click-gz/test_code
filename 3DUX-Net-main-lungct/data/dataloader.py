import os, glob
import torch, sys
from torch.utils.data import Dataset
import torch.utils.data as Data
import SimpleITK as sitk
import numpy as np

class DatasetsFlare(Data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = sorted(glob.glob(os.path.join(root_dir, 'imagesTr', '*.nii.gz')))
        self.labels = sorted(glob.glob(os.path.join(root_dir, 'labelsTr', '*.nii.gz')))
        self.root_dir = root_dir
        # for f in files:
        #     print(f)
    def __len__(self):
        # 返回数据集的大小
        return len(self.images)

    def __getitem__(self, index):
        train_image = self.images[index]
        train_label = self.labels[index]
        print("item: ", self.images[index])

        return train_image, train_label