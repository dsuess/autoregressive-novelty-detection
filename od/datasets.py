import functools as ft
import os
from collections import namedtuple
from operator import add
from pathlib import Path
from time import sleep

import mnist
import numpy as np
import torch
import torchvision as tv
from torch.utils.data import TensorDataset

from .utils import logger, isin
from PIL import Image

__all__ = ['Split', 'MNIST']


Split = namedtuple('Split', 'train, test')


class MNIST(tv.datasets.MNIST):
    def __init__(self, root, training_classes, train=True, transform=None, download=False):
        self.root = root
        self.transform = transform

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        path = self.training_file if train else self.test_file
        data, targets = torch.load(
            os.path.join(self.root, self.processed_folder, path))
        training_classes = torch.Tensor(list(training_classes)).to(targets.dtype)
        is_known = isin(targets, training_classes)

        if train:
            self.data = data[is_known.nonzero()][:, 0]
            self.targets = torch.ones(len(self.data)).to(targets.dtype)
        else:
            self.data = data
            self.targets = is_known

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of positive examples: {}'.format(self.targets.sum())
        return fmt_str

    @classmethod
    def load_split(cls, *args, transforms=None, **kwargs):
        transforms = list(transforms) if transforms is not None else []
        train_transform = tv.transforms.Compose(
            transforms + [tv.transforms.ToTensor()])
        test_transform = tv.transforms.Compose([tv.transforms.ToTensor()])
        datasets = Split(
            cls(*args, train=True, transform=train_transform, **kwargs),
            cls(*args, train=False, transform=test_transform, **kwargs))
        return datasets
