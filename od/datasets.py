import functools as ft
from operator import add
from pathlib import Path
import logging

from torch.utils.data.dataset import Dataset
from time import sleep
from PIL import Image


class UnsupervisedImageDataset(Dataset):

    def __init__(self, directory, extensions=['jpg'], transform=None):
        directory = Path(directory)
        self.paths = list(
            ft.reduce(add, [directory.glob(f'*.{ext}') for ext in extensions]))
        self.transform = transform
        logging.info(f'Found {len(self.paths)} images in {directory}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img
