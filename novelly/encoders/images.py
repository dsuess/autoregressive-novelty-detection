import numpy as np
import torch
from torch import nn

from novelly.utils import build_from_config

__all__ = ['ResidualAE']


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.residual_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels))

        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,
                      padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        """
        >>> x = torch.randn(1, 3, 32, 32)
        >>> block = EncoderBlock(3, 10)
        >>> tuple(block(x).shape)
        (1, 10, 16, 16)
        """
        return self.residual_path(x) + self.conv_path(x)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.residual_path = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1,
                               stride=2, output_padding=1),
            nn.BatchNorm2d(out_channels))

        self.conv_path = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        """
        >>> x = torch.randn(1, 3, 16, 16)
        >>> block = DecoderBlock(3, 10)
        >>> tuple(block(x).shape)
        (1, 10, 32, 32)
        """
        return self.residual_path(x) + self.conv_path(x)


def fc_layer(in_features, out_features, activation=None, batchnorm=True):
    layers = [nn.Linear(in_features, out_features)]

    if activation is not None:
        layers += [activation]
    if batchnorm:
        layers += [nn.BatchNorm1d(out_features)]

    return nn.Sequential(*layers)


class ResidualAE(nn.Module):

    def __init__(self, input_shape, encoder_sizes, fc_sizes, *, color_channels=3,
                 latent_activation=None, decoder_sizes=None):
        super().__init__()

        decoder_sizes = list(decoder_sizes) if decoder_sizes is not None \
            else list(reversed(encoder_sizes))

        conv_dims = list(zip([color_channels, *encoder_sizes], encoder_sizes))
        self.conv_encoder = nn.Sequential(*[EncoderBlock(*d) for d in conv_dims])

        self.input_shape = (color_channels, *input_shape)
        with torch.no_grad():
            self.conv_encoder.eval()
            dummy_input = torch.Tensor(1, *self.input_shape)
            dummy_output = self.conv_encoder(dummy_input)
        self.intermediate_size = dummy_output.shape[1:]
        self.first_fc_size = np.prod(self.intermediate_size)

        fc_dims = list(zip([self.first_fc_size, *fc_sizes], fc_sizes))
        self.fc_encoder = nn.Sequential(
            *[fc_layer(*d, activation=nn.LeakyReLU()) for d in fc_dims[:-1]],
            fc_layer(*fc_dims[-1], activation=latent_activation, batchnorm=False))
        self.embedding_dim = fc_sizes[-1]

        self.fc_decoder = nn.Sequential(
            *[fc_layer(d_out, d_in, activation=nn.LeakyReLU())
              for d_in, d_out in reversed(fc_dims)])
        conv_dims = list(zip(decoder_sizes, [*(decoder_sizes[1:]), color_channels]))
        self.conv_decoder = nn.Sequential(
            *[DecoderBlock(*d) for d in conv_dims],
            nn.Sigmoid())

    def encode(self, x):
        y = self.conv_encoder(x)
        y = y.view(-1, self.first_fc_size)
        y = self.fc_encoder(y)
        return y

    def decode(self, x):
        y = self.fc_decoder(x)
        y = y.view(-1, *self.intermediate_size)
        y = self.conv_decoder(y)
        return y

    def forward(self, x):
        """
        >>> model = ResidualAE((64, 64), [32], [5], color_channels=4)
        >>> x = torch.randn(2, 4, 64, 64)
        >>> y = model(x)
        >>> tuple(y.shape)
        (2, 4, 64, 64)
        """
        return self.decode(self.encode(x))

    @classmethod
    def from_config(cls, cfg):
        cfg = cfg.copy()
        if 'latent_activation' in cfg:
            cfg['latent_activation'] = build_from_config(nn, cfg.pop('latent_activation'))
        return cls(**cfg)
