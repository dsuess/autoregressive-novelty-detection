import logging
import os

import matplotlib.backends.backend_agg as pl_backend_agg
import matplotlib.pyplot as pl
import numpy as np
import torch
from torch import nn

__all__ = ['get_default_logger', 'logger']


def fc_layer(in_features, out_features, activation=None, batchnorm=True):
    layers = [nn.Linear(in_features, out_features)]

    if activation is not None:
        layers += [activation]
    if batchnorm:
        layers += [nn.BatchNorm1d(out_features)]

    return nn.Sequential(*layers)


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


logger = get_default_logger('AND')
