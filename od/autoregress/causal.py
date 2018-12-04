import numpy as np
import torch
from torch import nn

from od.utils import logger, mse_loss
from .linear import AutoregressiveLoss


__all__ = ['AutoregressiveVideoLoss', 'AutoregressiveConvLayer']


class AutoregressiveConv(nn.Conv1d):
    def __init__(self, dim, in_features, out_features, *, mask_type, **kwargs):
        # padding=1 for causal convolution
        super().__init__(dim * in_features, dim * out_features, kernel_size=2,
                         padding=1, groups=dim, **kwargs)
        assert mask_type in {'A', 'B'}
        self.dim = dim
        self.in_features = in_features
        self.out_features = out_features
        self.mask_type = mask_type

        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)
        mask = self.mask.view(dim, out_features, in_features, 2)
        for j in range(dim):
            mask[j, :, j + int(mask_type == 'B'):, 1] = 0

    def forward(self, x):
        """
        Input Shape: (batch, dim, in_features, time)

        >>> layer = AutoregressiveConv(5, 10, 20, mask_type='A')
        >>> x = torch.rand(1, 5, 10, 4)
        >>> y = layer(x)
        >>> tuple(y.shape)
        (1, 5, 20, 4)
        """
        y = x.view(x.size(0), self.dim * self.in_features, x.size(3))
        y = nn.functional.conv1d(y, self.mask * self.weight, self.bias,
                                 stride=self.stride, padding=self.padding,
                                 dilation=self.dilation, groups=self.groups)
        # Crop padded time axis to make causal
        y = y[..., :x.size(3)]
        return y.view(x.size(0), self.dim, self.out_features, x.size(3))

    @staticmethod
    def _get_name():
        return 'AutoregressiveConv'

    def extra_repr(self):
        return f'dim={self.dim}, in_features={self.in_features}, ' \
            f'out_features={self.out_features}, mask_type={self.mask_type}'


class AutoregressiveConvLayer(nn.Module):
    def __init__(self, *args, activation=None, batchnorm=True, **kwargs):
        super().__init__()
        self.conv = AutoregressiveConv(*args, **kwargs)
        self.activation = activation() if activation is not None else None

        if batchnorm:
            size = self.conv.dim * self.conv.out_features
            self.batchnorm = nn.BatchNorm1d(size, track_running_stats=False)
        else:
            self.batchnorm = None

    def forward(self, x):
        """
        >>> layer = AutoregressiveConvLayer(5, 10, 20, mask_type='A', activation=nn.ReLU)
        >>> x = torch.rand(1, 5, 10, 4)
        >>> y = layer(x)
        >>> tuple(y.shape)
        (1, 5, 20, 4)
        """
        y = self.conv(x)
        if self.activation is not None:
            y = self.activation(y)
        if self.batchnorm is not None:
            shape = y.shape
            y = y.view(shape[0], shape[1] * shape[2], shape[3])
            y = self.batchnorm(y)
            y = y.view(shape[0], shape[1], shape[2], shape[3])
        return y


class AutoregressiveVideoLoss(AutoregressiveLoss):

    def _autoreg_loss(self, latent):
        # move time axis to back as expected by regressor
        # FIXME Change order in autoencoder to have time axis last
        latent = latent.permute(0, 2, 1)
        latent_binned = (latent * self.bins.float()).type(torch.int64)
        latent_binned = latent_binned.clamp(0, self.bins - 1)
        latent_binned = latent_binned
        latent_binned_pred = self.regressor(
            latent.view(latent.size(0), latent.size(1), 1, latent.size(2)))
        latent_binned_pred = latent_binned_pred.permute(0, 2, 1, 3)

        autoreg_loss = nn.functional.cross_entropy(
            latent_binned_pred, latent_binned, reduction='none')
        return autoreg_loss.mean(dim=2).mean(dim=1)
