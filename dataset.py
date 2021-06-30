import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import get_label


class LiDARDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir="dataset/training",
        dataset="KITTI_CAR",
        crop=True,
        num_point=12000,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.dataset = dataset
        self.crop = crop
        self.num_point = num_point
        self.velo_dir = os.path.join(self.root_dir, "velodyne_reduced")
        self.label_dir = os.path.join(self.root_dir, "label_2")
        self.fileID = [fname.split(".")[0] for fname in os.listdir(self.velo_dir)]

    def __len__(self):
        return len(os.listdir(self.velo_dir))

    def __getitem__(self, idx):
        points = np.reshape(
            np.fromfile(os.path.join(self.velo_dir, self.fileID[idx]) + ".bin", "<f4"),
            (-1, 4),
        )[..., :3]
        if self.crop:
            x,y,z = points.T
            index = np.intersect1d(np.where(abs(x)<=60),np.where(np.logical_and(y<=60,y>=-60)))
            index = np.intersect1d(np.where(np.logical_and(z<=3,z>=-3)),index)
            points = points[index]
        label = get_label(
            os.path.join(self.label_dir, self.fileID[idx] + ".txt"), self.dataset
        )
        return torch.from_numpy(points[: self.num_point]).float(), torch.Tensor(label)
