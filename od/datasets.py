import functools as ft
import logging
from operator import add
from pathlib import Path
from time import sleep

import mnist
import numpy as np
import torch
from torch.utils.data import TensorDataset


def UnsupervisedMNIST(*, exclude_digits=None, **kwargs):
    exclude_digits = set() if exclude_digits is not None \
        else exclude_digits

    images = np.concatenate((mnist.train_images(), mnist.test_images()))
    labels = np.concatenate((mnist.train_labels(), mnist.test_labels()))
    exclude = np.isin(labels, exclude_digits)
    images = torch.Tensor(images[~exclude][..., None])
    return TensorDataset(images)
