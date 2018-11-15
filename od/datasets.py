import functools as ft
import logging
from operator import add
from pathlib import Path
from time import sleep

import mnist
import numpy as np
import torch
from torch.utils.data import TensorDataset


__all__ = ['UnsupervisedMNIST']


def UnsupervisedMNIST(*, exclude_digits=None):
    exclude_digits = set() if exclude_digits is not None \
        else exclude_digits

    # we have to normalize to [0, 1] since output layer uses sigmoid
    images = np.concatenate((mnist.train_images(), mnist.test_images())) / 255

    labels = np.concatenate((mnist.train_labels(), mnist.test_labels()))
    exclude = np.isin(labels, exclude_digits)
    images = torch.Tensor(images[~exclude][..., None]).permute(0, 3, 1, 2)

    return TensorDataset(images)
