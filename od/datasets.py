import functools as ft
from operator import add
from pathlib import Path
from time import sleep
from collections import namedtuple

import mnist
import numpy as np
import torch
from torch.utils.data import TensorDataset

from .utils import logger

__all__ = ['Split', 'mnist_novelty_dataset']


Split = namedtuple('Split', 'train, test')


def mnist_novelty_dataset(*, novel_digits=None):
    # we have to normalize to [0, 1] since output layer uses sigmoid
    images = Split(mnist.train_images() / 255, mnist.test_images() / 255)
    labels = Split(mnist.train_labels(), mnist.test_labels())

    novel_digits = set() if novel_digits is None else novel_digits
    novel_digits = np.array(list(novel_digits), dtype=labels.train.dtype)
    exclude = np.isin(labels.train, novel_digits)

    images = Split(
        train=images.train[~exclude, ][:, None],
        test=np.concatenate([images.train[exclude], images.test])[:, None])

    is_known_test = ~np.isin(labels.test, novel_digits)
    is_known_test = np.concatenate([np.zeros(sum(exclude), dtype=bool), is_known_test])
    is_known = Split(
        train=np.ones(len(images.train), dtype=np.uint8),
        test=is_known_test.astype(np.uint8))

    result = Split(
        train=TensorDataset(torch.Tensor(images.train), torch.Tensor(is_known.train)),
        test=TensorDataset(torch.Tensor(images.test), torch.Tensor(is_known.test)))
    logger.info('Created dataset with:')
    logger.info(f'    - {len(result.train)} training examples')
    logger.info(f'    - {len(result.test)} test examples')
    logger.info(f'    - {len(result.train) + np.sum(is_known_test)} known examples')
    logger.info(f'    - {np.sum(~is_known_test)} unknown examples')
    return result
