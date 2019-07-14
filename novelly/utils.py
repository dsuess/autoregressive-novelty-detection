import itertools as it
import logging
import os
from collections import namedtuple
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

import matplotlib.backends.backend_agg as pl_backend_agg
import matplotlib.pyplot as pl

__all__ = ['get_default_logger', 'logger', 'build_from_config', 'get_model_module']


Split = namedtuple('Split', 'train, valid')


def get_default_logger(name):
    logger = logging.getLogger(name)
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))
    return logger


def render_dict_as_table(the_dict):
    lines = ['Key | Value', '--- | ---']
    lines += [f'{key} | {val}' for key, val in the_dict.items()]
    return lines


def sample_distribution_overlap(xs1, xs2, nr_bins=200):
    vmin = min(np.min(xs1), np.min(xs2))
    vmax = max(np.max(xs1), np.max(xs2))
    bins = np.linspace(vmin, vmax, nr_bins)
    y1, _ = np.histogram(xs1, bins=bins, density=True)
    y2, _ = np.histogram(xs2, bins=bins, density=True)
    return np.sum(np.min([y1, y2], axis=0)) * (vmax - vmin ) / nr_bins


def render_mpl_figure(figure=None, close_figure=True, channel_order='CHW'):
    assert channel_order in {'CHW', 'HWC'}
    figure = figure if figure is not None else pl.gcf()
    canvas = pl_backend_agg.FigureCanvasAgg(figure)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = figure.canvas.get_width_height()
    image = data.reshape([h, w, 4])[:, :, 0:3]

    if channel_order == 'CHW':
        image = image.transpose(2, 0, 1)

    if close_figure:
        pl.close(figure)

    return image


def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)


def interleave(*args):
    """
    >>> list(interleave([1, 2, 3], [4, 5, 6]))
    [1, 4, 2, 5, 3, 6]
    """
    return it.chain(*zip(*args))


def connected_compoents(arr):
    end = -1
    while end < len(arr) - 1:
        idx, = np.nonzero(arr[end + 1:])
        try:
            start = idx[0] + end + 1
        except IndexError:
            return

        idx, = np.nonzero(~arr[start + 1:])
        try:
            end = idx[0] + start + 1
        except IndexError:
            end = len(arr) - 1
        yield start, end


def iterbatch(iterable, batch_size=None):
    if batch_size is None:
        yield [iterable]
    else:
        iterator = iter(iterable)
        try:
            while True:
                first_elem = next(iterator)
                yield it.chain((first_elem,),
                               it.islice(iterator, batch_size - 1))
        except StopIteration:
            pass


def get_from_modules(name, modules):
    for mod in modules:
        try:
            return getattr(mod, name)
        except AttributeError:
            pass
    raise AttributeError(f'{name} not found in {modules}')


def build_from_config(module, cfg, **kwargs):
    cfg = cfg.copy()
    name = cfg.pop('type')
    module = list(module) if isinstance(module, Sequence) else [module]
    the_class = get_from_modules(name, module)

    try:
        config_fn = the_class.from_config
    except AttributeError:
        return the_class(**cfg, **kwargs)
    else:
        return config_fn(cfg, **kwargs)


def get_model_module(model):
    try:
        return model.module
    except AttributeError:
        return model



logger = get_default_logger('novelly')
