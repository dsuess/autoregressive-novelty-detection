from torch import nn
from .linear import AutoregressiveLayer


__all__ = ['AutoregresionModule']


class AutoregresionModule(nn.Module):
    def __init__(self, dim, layer_sizes, activation=nn.LeakyReLU,
                 batchnorm=True, layer=AutoregressiveLayer):
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        dimensions = list(zip([1] + self.layer_sizes, self.layer_sizes))
        activation_fns = [activation] * (len(dimensions) - 1) + [None]
        batchnorm = [batchnorm] * (len(dimensions) - 1) + [False]
        mask_types = ['A'] + ['B'] * (len(dimensions) - 1)
        configuration = zip(dimensions, activation_fns, mask_types, batchnorm)

        self.layers = nn.Sequential(
            *[layer(dim, d_in, d_out, activation=fn, batchnorm=bn, mask_type=mt)
              for (d_in, d_out), fn, mt, bn in configuration])

    @property
    def bins(self):
        return self.layer_sizes[-1]

    def forward(self, x):
        return self.layers(x)
