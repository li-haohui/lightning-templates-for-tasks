import abc
import torch

import pandas as pd
import open3d as o3d
import numpy as np
import os

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data_info=None, data_dir=None, voxel_size=5):
        super().__init__()

        self.data_dir = data_dir
        self.voxel_size = voxel_size

        self.INFO = pd.read_csv(data_info).to_numpy()
        print("Sucessfully load {}!".format(data_info))

    @abc.abstractmethod
    def _get_info(self, index):
        pass

    def __len__(self):
        return self.INFO.shape[0]


class QADataset(BaseDataset):
    def __init__(self, data_info=None, data_dir=None, voxel_size=5):
        super().__init__(data_info, data_dir, voxel_size)

    def __getitem__(self, index):
        name, mos = self.INFO[index, 0], self.INFO[index, 1]
        ply = o3d.io.read_point_cloud(os.path.join(self.data_dir, name))


        coords = np.array(ply.points, dtype=np.float32),
        feats = np.array(ply.colors, dtype=np.float32),

        return {
            "coord": coords,
            "feats": feats,
            "lables": mos
        }