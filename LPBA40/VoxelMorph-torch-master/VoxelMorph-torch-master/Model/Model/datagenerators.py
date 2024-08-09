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
        random.shuffle(tar_list)
        tar_file = tar_list[0]
        x = sitk.GetArrayFromImage(sitk.ReadImage(path))
        x_seg = sitk.GetArrayFromImage(sitk.ReadImage(path.replace("imagesTs", "masksTs")))

        y = sitk.GetArrayFromImage(sitk.ReadImage(tar_file))
        y_seg = sitk.GetArrayFromImage(sitk.ReadImage(tar_file.replace("imagesTs", "masksTs")))

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class LPBA40Datasets(Dataset):
    def __init__(self, data_path):
        self.paths = data_path

    def __getitem__(self, index):
        img = sitk.GetArrayFromImage(sitk.ReadImage(self.paths[index]))[np.newaxis, np.newaxis, ...]
        img =  (np.max(img) - img) / (np.max(img) - np.min(img))
        img = torch.from_numpy(img)
        return img

    def __len__(self):
        return len(self.paths)

class LPBA40InfDatasets(Dataset):
    def __init__(self, data_path):
        self.paths = data_path

    def __getitem__(self, index):
        img = sitk.GetArrayFromImage(sitk.ReadImage(self.paths[index]))[np.newaxis, ...]
        img = (np.max(img) - img) / (np.max(img) - np.min(img))
        img = torch.from_numpy(img)
        seg_path = self.paths[index].replace("test", "label").replace(".delineation.skullstripped", ".delineation.structure.label")
        seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))[np.newaxis, ...]
        seg = torch.from_numpy(seg)
        return img, seg

    def __len__(self):
        return len(self.paths)

class LPBA40InfOriDatasets(Dataset):
    def __init__(self, data_path):
        self.paths = data_path

    def __getitem__(self, index):
        img = sitk.GetArrayFromImage(sitk.ReadImage(self.paths[index]))[np.newaxis, ...]
        nimg = (np.max(img) - img)/(np.max(img) - np.min(img))
        img, nimg = torch.from_numpy(img), torch.from_numpy(nimg)
        seg_path = self.paths[index].replace("test", "label").replace(".delineation.skullstripped", ".delineation.structure.label")
        seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))[np.newaxis, ...]
        seg = torch.from_numpy(seg)
        return nimg, seg, img

    def __len__(self):
        return len(self.paths)