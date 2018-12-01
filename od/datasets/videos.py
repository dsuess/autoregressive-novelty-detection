import functools as ft
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class NoveltyVideoDataset(Dataset):

    def __init__(self, frames_dir, frame_mask_path, *, window, step=1,
                 file_format='{:03d}.jpg'):
        self.frames_dir = Path(frames_dir)
        self.target = 1 - np.load(frame_mask_path)
        self.window = window
        self.step = step
        self.file_format = file_format

    def verify_path(self):
        for i in range(len(self.frame_mask)):
            path = self.frames_dir / self.file_format.format(i)
            if not path.exists():
                raise ValueError(f'Path to {path} not found')
        return self

    def __len__(self):
        return (len(self.target) - self.window) // self.step + 1

    # TODO Make this dependent on window size
    @ft.lru_cache(maxsize=16)
    def _load_img(self, idx):
        path = self.frames_dir / self.file_format.format(idx)
        return np.array(Image.open(path))

    def __getitem__(self, i):
        indices = np.arange(self.window) + self.step * i
        images = np.array([self._load_img(i) for i in indices])
        target = self.target[indices]
        return images, target
