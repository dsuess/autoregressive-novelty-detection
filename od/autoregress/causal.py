import numpy as np
import torch
from torch import nn

from od.utils import logger, mse_loss

#  __all__ = ['AutoregresionModule', 'AutoregressiveLoss']


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
        return 'AutoregressiveLinear'

    def extra_repr(self):
        return f'dim={self.dim}, in_features={self.in_features}, ' \
            f'out_features={self.out_features}, mask_type={self.mask_type}'


