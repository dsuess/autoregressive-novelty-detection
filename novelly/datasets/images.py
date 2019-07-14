import abc
import functools as ft
import os
import pickle
import tempfile
from collections import namedtuple
from operator import add
from pathlib import Path

import numpy as np
import torch
import torchvision as tv
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset

from novelly.utils import isin, logger, build_from_config

__all__ = ['MNIST', 'CIFAR10']


class NoveltyDetectionDataset(Dataset, abc.ABC):

    MODE = None
    LOADER = DataLoader

    def __init__(self, positive_classes, root=None, train=True, transform=None,
                 download=False):
        self.root = root if root is not None \
            else Path(tempfile.gettempdir()) / type(self).__name__
        self.transform = transform
        self.yield_target = not train

        if download:
            self.root.mkdir(exist_ok=True)
            self.download()

        data, targets = self.load(train)
        positive_classes = torch.Tensor(list(positive_classes)).to(targets.dtype)
        is_known = isin(targets, positive_classes)

        if train:
            self.data = data[is_known.nonzero()][:, 0]
            self.targets = torch.ones(len(self.data)).to(targets.dtype)
        else:
            self.data = data
            self.targets = is_known

    @abc.abstractmethod
    def load(self, train):
        pass

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of positive examples: {}'.format(self.targets.sum())
        return fmt_str

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode=self.MODE)

        if self.transform is not None:
            img = self.transform(img)

        return (img, target) if self.yield_target else img

    @classmethod
    def from_config(cls, cfg, **kwargs):
        cfg = cfg.copy()

        try:
            transforms = cfg.pop('transforms')
        except KeyError:
            transforms = []
        else:
            transforms = [
                build_from_config(tv.transforms, c) for c in transforms]

        transforms.append(tv.transforms.ToTensor())
        transform = tv.transforms.Compose(transforms)
        return cls(**cfg, transform=transform, **kwargs)



class MNIST(NoveltyDetectionDataset, tv.datasets.MNIST):

    MODE = 'L'
    def load(self, train):
        path = self.training_file if train else self.test_file
        return torch.load(os.path.join(str(self.root), self.processed_folder, path))


class CIFAR10(NoveltyDetectionDataset, tv.datasets.CIFAR10):

    MODE = 'RGB'

    def load(self, train):
        if train:
            data = []
            targets = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                entry = pickle.load(fo, encoding='latin1')
                data.append(entry['data'])
                if 'labels' in entry:
                    targets += entry['labels']
                else:
                    targets += entry['fine_labels']
                fo.close()
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            entry = pickle.load(fo, encoding='latin1')
            data = entry['data']
            if 'labels' in entry:
                targets = entry['labels']
            else:
                targets = entry['fine_labels']
            fo.close()

        data = np.concatenate(data)
        data = data.reshape((-1, 3, 32, 32))
        data = data.transpose((0, 2, 3, 1))  # convert to HWC
        targets = np.array(targets)
        return torch.from_numpy(data), torch.from_numpy(targets)
