import os, glob
import torch, sys
from torch.utils.data import Dataset
import SimpleITK as sitk
import matplotlib.pyplot as plt
import random
import numpy as np


class OASISBrainDataset(Dataset):
    def __init__(self, data_path):
        self.paths = data_path

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        tar_list = self.paths.copy()
        tar_list.remove(path)
        random.shuffle(tar_list)
        tar_file = tar_list[0]
        x = sitk.GetArrayFromImage(sitk.ReadImage(path))
        x_seg = sitk.GetArrayFromImage(sitk.ReadImage(path.replace("imagesTr", "masksTr")))

        y = sitk.GetArrayFromImage(sitk.ReadImage(tar_file))
        y_seg = sitk.GetArrayFromImage(sitk.ReadImage(tar_file.replace("imagesTr", "masksTr")))

        x, y = x[None, ...], y[None, ...]
        x = (x - np.min(x))/(np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class OASISBrainInferDataset(Dataset):
    def __init__(self, data_path):
        self.paths = data_path

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        tar_list = self.paths.copy()
        tar_list.remove(path)
        # random.shuffle(tar_list)
        tar_file = tar_list[0]
        x = sitk.GetArrayFromImage(sitk.ReadImage(path))
        x_seg = sitk.GetArrayFromImage(sitk.ReadImage(path.replace("imagesTs", "masksTs")))

        y = sitk.GetArrayFromImage(sitk.ReadImage(tar_file))
        y_seg = sitk.GetArrayFromImage(sitk.ReadImage(tar_file.replace("imagesTs", "masksTs")))

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        print(path, tar_file)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)