from collections import namedtuple

import numpy as np
import torch
from torch import nn

from novelly.utils import build_from_config, logger


__all__ = ['AutoregressiveLayer', 'AutoregressionModule']


# TODO Optimize axis alignment to get rid of permute in loss
class AutoregressiveLinear(nn.Linear):
    def __init__(self, dim, in_features, out_features, *, mask_type, **kwargs):
        super().__init__(dim * in_features, dim * out_features, **kwargs)
        assert mask_type in {'A', 'B'}
        self.dim = dim
        self.in_features = in_features
        self.out_features = out_features
        self.mask_type = mask_type

        self.in_shape = (dim * in_features,)
        self.out_shape = (dim, out_features)

        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)
        mask = self.mask.view(dim, out_features, dim, in_features)
        for j in range(dim):
            mask[j, :, j + int(mask_type == 'B'):] = 0

    def forward(self, x):
        y = x.view(-1, *self.in_shape)
        y = nn.functional.linear(y, self.mask * self.weight, self.bias)
        return y.view(-1, *self.out_shape)

    @staticmethod
    def _get_name():
        return 'AutoregressiveLinear'

    def extra_repr(self):
        extra_repr = super().extra_repr()
        return f'dim={self.dim}, {extra_repr}, mask_type={self.mask_type}'


class AutoregressiveLayer(nn.Module):
    def __init__(self, *args, activation=None, batchnorm=True, **kwargs):
        super().__init__()
        self.linear = AutoregressiveLinear(*args, **kwargs)
        self.activation = activation
        self.in_shape = (np.prod(self.linear.out_shape),)
        self.out_shape = self.linear.out_shape

        if batchnorm:
            # TODO Find out why track_running_stats=True doesn't work well here
            self.batchnorm = nn.BatchNorm1d(*self.in_shape, track_running_stats=False)
        else:
            self.batchnorm = None

    def forward(self, x):
        y = self.linear(x)
        if self.activation is not None:
            y = self.activation(y)
        if self.batchnorm is not None:
            y = y.view(-1, *self.in_shape)
            y = self.batchnorm(y)
            y = y.view(-1, *self.out_shape)
        return y


class AutoregressionModule(nn.Module):
    def __init__(self, autoencoder, layer_sizes, activation=None,
                 batchnorm=True, layer=AutoregressiveLayer):
        super().__init__()
        self.autoencoder = autoencoder
        self.layer_sizes = list(layer_sizes)
        dimensions = list(zip([1] + self.layer_sizes, self.layer_sizes))
        activation = activation if activation is not None else nn.LeakyReLU()
        activation_fns = [activation] * (len(dimensions) - 1) + [None]
        batchnorm = [batchnorm] * (len(dimensions) - 1) + [False]
        mask_types = ['A'] + ['B'] * (len(dimensions) - 1)
        configuration = zip(dimensions, activation_fns, mask_types, batchnorm)

        self.regressor = nn.Sequential(
            *[layer(autoencoder.embedding_dim, d_in, d_out, activation=fn,
                    batchnorm=bn, mask_type=mt)
              for (d_in, d_out), fn, mt, bn in configuration])
        self.register_scalar('bins', self.layer_sizes[-1], torch.int64)

    def register_scalar(self, name, val, dtype):
        val = torch.Tensor([val]).reshape(tuple()).to(dtype)
        self.register_buffer(name, val)

    def forward(self, x, return_reconstruction=False):
        latent = self.autoencoder.encode(x)
        latent_binned = (latent * self.bins.float()).type(torch.int64)
        latent_binned = latent_binned.clamp(0, self.bins - 1)
        latent_binned_pred = self.regressor(latent).permute(0, 2, 1)
        autoreg_score = nn.functional.cross_entropy(
            latent_binned_pred, latent_binned, reduction='none').mean(dim=1)
        return autoreg_score if not return_reconstruction \
            else (autoreg_score, self.autoencoder.decode(latent))

    @classmethod
    def from_config(cls, cfg, **kwargs):
        cfg = cfg.copy()
        if 'latent_activation' in cfg:
            cfg['activation'] = build_from_config(nn, cfg.pop('latent_activation'))
        return cls(**cfg, **kwargs)
