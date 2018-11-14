import functools as ft
import logging
from operator import add
from pathlib import Path
from time import sleep

import mnist
import numpy as np
from torch.utils.data.dataset import Dataset


class UnsupervisedMNIST(Dataset):

    def __init__(self, exclude_digits=None, transform=None):
        self.exclude_digits = set() if exclude_digits is not None \
            else exclude_digits

        images = np.concatenate((mnist.train_images(), mnist.test_images()))
        labels = np.concatenate((mnist.train_labels(), mnist.test_labels()))
        exclude = np.isin(labels, exclude_digits)
        self.images = images[~exclude][..., None]
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)
