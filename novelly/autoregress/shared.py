from torch import nn

from novelly.utils import build_from_config

from .linear import AutoregressiveLayer

__all__ = ['AutoregressionModule']


class AutoregressionModule(nn.Module):
    def __init__(self, embedding_dim, layer_sizes, activation=None,
                 batchnorm=True, layer=AutoregressiveLayer):
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        dimensions = list(zip([1] + self.layer_sizes, self.layer_sizes))
        activation = activation if activation is not None else nn.LeakyReLU()
        activation_fns = [activation] * (len(dimensions) - 1) + [None]
        batchnorm = [batchnorm] * (len(dimensions) - 1) + [False]
        mask_types = ['A'] + ['B'] * (len(dimensions) - 1)
        configuration = zip(dimensions, activation_fns, mask_types, batchnorm)

        self.layers = nn.Sequential(
            *[layer(embedding_dim, d_in, d_out, activation=fn, batchnorm=bn, mask_type=mt)
              for (d_in, d_out), fn, mt, bn in configuration])

    @property
    def bins(self):
        return self.layer_sizes[-1]

    def forward(self, x):
        return self.layers(x)

    @classmethod
    def from_config(cls, cfg):
        cfg = cfg.copy()
        if 'latent_activation' in cfg:
            cfg['activation'] = build_from_config(nn, cfg.pop('latent_activation'))
        return cls(**cfg)
