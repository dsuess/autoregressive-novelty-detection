import torch
from torch import nn
from collections import namedtuple


# TODO Optimize axis alignment to get rid of permute in loss
class AutoregressiveLayer(nn.Linear):
    def __init__(self, dim, in_features, out_features, *, mask_type,
                 batch_norm=True, activation=None, **kwargs):
        super().__init__(dim * in_features, dim * out_features, **kwargs)
        assert mask_type in {'A', 'B'}
        self.dim = dim
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(dim * out_features)
        else:
            self.batch_norm = lambda x: x

        self.in_shape = (dim * in_features,)
        self.out_shape = (dim, out_features)

        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)
        mask = self.mask.view(dim, out_features, dim, in_features)
        for j in range(dim):
            mask[j, :, j + int(mask_type == 'B'):] = 0


    def forward(self, x):
        self.weight.data *= self.mask
        y = super().forward(x.view(-1, *self.in_shape))
        y = self.batch_norm(y)

        if self.activation is not None:
            y = self.activation(y)

        return y.view(-1, *self.out_shape)

    def __repr__(self):
        return f'AutoregressiveLayer(dim={self.dim}, ' \
            f'in_features={self.in_features}, ' \
            f'out_features={self.out_features})'


class AutoregresionModule(nn.Module):
    def __init__(self, dim, mfc_layers, **kwargs):
        super().__init__(**kwargs)
        self.mfc_layers = list(mfc_layers)
        dimensions = list(zip([1] + self.mfc_layers, self.mfc_layers[1:]))
        activation_fns = [nn.functional.leaky_relu] * (len(dimensions) - 1) + [None]
        mask_types = ['A'] + ['B'] * (len(dimensions) - 1)
        configuration = zip(dimensions, activation_fns, mask_types)

        self.layers = nn.Sequential(
            *[AutoregressiveLayer(dim, d_in, d_out, activation=fn, mask_type=mt)
              for (d_in, d_out), fn, mt in configuration])

    @property
    def bins(self):
        return self.mfc_layers[-1]

    def forward(self, x):
        return self.layers(x)


class AutoregressiveLoss(nn.Module):

    Result = namedtuple('Result', 'reconstruction, autoregressive')

    def __init__(self, autoencoder, autoreg, **kwargs):
        super().__init__(**kwargs)
        self.autoencoder = autoencoder
        self.autoreg = autoreg

    def forward(self, x):
        latent = nn.functional.sigmoid(self.autoencoder.encode(x))
        reconstruction = self.autoencoder.decode(latent)
        reconstruction_loss = nn.functional.mse_loss(x, reconstruction)

        latent_binned = (latent * self.autoreg.bins).type(torch.int64)
        latent_binned_pred = self.autoreg(latent).permute(0, 2, 1)
        autoreg_loss = nn.functional.cross_entropy(
            latent_binned_pred, latent_binned)

        return self.Result(reconstruction_loss, autoreg_loss)
