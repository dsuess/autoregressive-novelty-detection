from collections import namedtuple

import numpy as np
import torch
from torch import nn

from .utils import logger

__all__ = ['AutoregresionModule', 'AutoregressiveLoss']


# TODO Optimize axis alignment to get rid of permute in loss
class AutoregressiveLinear(nn.Linear):
    def __init__(self, dim, in_features, out_features, *, mask_type,
                 batchnorm=True, activation=None, **kwargs):
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

    def __repr__(self):
        return f'AutoregressiveLinear(dim={self.dim}, ' \
            f'in_features={self.in_features}, ' \
            f'out_features={self.out_features}, ' \
            f'mask_type={self.mask_type})'


class AutoregressiveLayer(nn.Module):
    def __init__(self, *args, activation=None, batchnorm=True, **kwargs):
        super().__init__()
        self.linear = AutoregressiveLinear(*args, **kwargs)
        self.activation = activation() if activation is not None else None
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


class AutoregresionModule(nn.Module):
    def __init__(self, dim, mfc_layers, activation=nn.LeakyReLU,
                 batchnorm=True):
        super().__init__()
        self.mfc_layers = list(mfc_layers)
        dimensions = list(zip([1] + self.mfc_layers, self.mfc_layers))
        activation_fns = [activation] * (len(dimensions) - 1) + [None]
        batchnorm = [batchnorm] * (len(dimensions) - 1) + [False]
        mask_types = ['A'] + ['B'] * (len(dimensions) - 1)
        configuration = zip(dimensions, activation_fns, mask_types, batchnorm)

        self.layers = nn.Sequential(
            *[AutoregressiveLayer(dim, d_in, d_out, activation=fn,
                                  batchnorm=bn, mask_type=mt)
              for (d_in, d_out), fn, mt, bn in configuration])

    @property
    def bins(self):
        return self.mfc_layers[-1]

    def forward(self, x):
        return self.layers(x)


class AutoregressiveLoss(nn.Module):

    Result = namedtuple('Result', 'reconstruction, autoregressive')

    def __init__(self, encoder, regressor, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.regressor = regressor

        self.register_scalar('bins', self.regressor.bins, torch.int64)

    def register_scalar(self, name, val, type):
        val = torch.Tensor([val]).reshape(tuple()).to(type)
        self.register_buffer(name, val)

    def _autoreg_loss(self, latent):
        latent_binned = (latent * self.bins.float()).type(torch.int64)
        latent_binned = latent_binned.clamp(0, self.bins - 1)
        latent_binned_pred = self.regressor(latent).permute(0, 2, 1)
        autoreg_loss = nn.functional.cross_entropy(
            latent_binned_pred, latent_binned, reduction='none')
        return autoreg_loss.mean(dim=1)

    def forward(self, x):
        latent = self.encoder.encode(x)
        autoreg_loss = self._autoreg_loss(latent)

        reconstruction = self.encoder.decode(latent)
        reconstruction_loss = nn.functional.mse_loss(
            x, reconstruction, reduction='none')
        reconstruction_loss = reconstruction_loss.view(reconstruction_loss.size(0), -1)
        reconstruction_loss = reconstruction_loss.sum(dim=1)

        return self.Result(reconstruction_loss, autoreg_loss)

    def predict(self, x):
        with torch.no_grad():
            latent = self.encoder.encode(x)
            return self._autoreg_loss(latent)
