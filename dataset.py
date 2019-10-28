import torch
import cv2
import os
import numpy as np
from torch.utils.data.dataset import Dataset
from config import config
np.random.seed(2)

class GOT_10KDataset(Dataset):
    def __init__(self, data_dir, z_transforms, x_transforms, training=True):
        self.data_dir = data_dir
        self.videos = os.listdir(data_dir)
        self.z_transforms = z_transforms
        self.x_transforms = x_transforms
        if training:
            self.num = config.train_per_epoch
        else:
            self.num = config.val_per_epoch


    def __getitem__(self, index):
        index = index % len(self.videos)
        video = self.videos[index]
        video_path = os.path.join(self.data_dir, video)
        n_frames = len(os.listdir(video_path))
        z_id = np.random.choice(n_frames)
        z_path = os.path.join(video_path, "{:0>8d}.x.jpg".format(z_id+1))
        z = cv2.imread(z_path, cv2.IMREAD_COLOR)
        z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        low_limit = max(0, z_id - config.frame_range)
        up_limit = min(n_frames, z_id + config.frame_range)
        x_id = np.random.choice(range(low_limit,up_limit))
        x_path = os.path.join(video_path, "{:0>8d}.x.jpg".format(x_id+1))
        x = cv2.imread(x_path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        if np.random.rand(1) < config.gray_ratio: # data augmentation for gray image to color image
            z = cv2.cvtColor(z, cv2.COLOR_RGB2GRAY)
            z = cv2.cvtColor(z, cv2.COLOR_GRAY2RGB)
            x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        z = self.z_transforms(z)
        x = self.x_transforms(x)
        return z, x

    def __len__(self):
        return self.num
