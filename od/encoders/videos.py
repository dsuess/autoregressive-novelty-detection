import torch
from torch import nn
import numpy as np


__all__ = ['ResidualVideoAE']


class CausalConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, padding=None,
                 **kwargs):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        time_ks, *_ = kernel_size
        assert time_ks > 1, 'For kernel_size=1 use normal nn.Conv3d'

        if isinstance(padding, int):
            padding = (padding, padding)
        space_padding = padding if padding is not None else (0, 0)
        padding = (time_ks - 1, *space_padding)

        super().__init__(in_channels, out_channels, kernel_size,
                         padding=padding, **kwargs)

    @staticmethod
    def _get_name():
        return 'CausalConv3d'

    def forward(self, x):
        """
        >>> layer = CausalConv3d(2, 2, (4, 3, 3))
        >>> x = torch.rand((1, 2, 5, 3, 3))
        >>> tuple(layer(x).shape)
        (1, 2, 5, 1, 1)

        >>> layer = CausalConv3d(2, 2, (3, 3, 3), stride=2)
        >>> x = torch.rand((1, 2, 4, 3, 3))
        >>> tuple(layer(x).shape)
        (1, 2, 2, 1, 1)

        """
        y = super().forward(x)
        chomp, *_ = self.padding
        stride, *_ = self.stride
        return y[:, :, :-chomp // stride].contiguous()


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, temporal_stride=2):
        super().__init__()

        self.residual_path = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1,
                      stride=(temporal_stride, 2, 2)),
            nn.BatchNorm3d(out_channels))

        self.conv_path = nn.Sequential(
            CausalConv3d(in_channels, out_channels, kernel_size=3,
                         stride=(temporal_stride, 2, 2), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(out_channels),
            CausalConv3d(out_channels, out_channels, kernel_size=3, stride=1,
                         padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(out_channels),
            CausalConv3d(out_channels, out_channels, kernel_size=3, stride=1,
                         padding=1),
            nn.BatchNorm3d(out_channels))

    def forward(self, x):
        """
        >>> x = torch.randn(1, 3, 8, 32, 32)
        >>> block = EncoderBlock(3, 10)
        >>> tuple(block(x).shape)
        (1, 10, 4, 16, 16)
        """
        return self.residual_path(x) + self.conv_path(x)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, temporal_stride=2):
        super().__init__()

        self.residual_path = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1,
                               stride=(temporal_stride, 2, 2), output_padding=1),
            nn.BatchNorm3d(out_channels))

        self.conv_path = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3,
                               stride=(temporal_stride, 2, 2), padding=1,
                               output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm3d(out_channels))

    def forward(self, x):
        """
        >>> x = torch.randn(1, 3, 4, 16, 16)
        >>> block = DecoderBlock(3, 10)
        >>> tuple(block(x).shape)
        (1, 10, 8, 32, 32)
        """
        return self.residual_path(x) + self.conv_path(x)


class ParallelBatchNorm1d(nn.BatchNorm1d):

    def forward(self, x):
        x_reshaped = x.view(-1, x.shape[2])
        y_reshaped = super().forward(x_reshaped)
        y = y_reshaped.view(*x.shape)
        return y

    @staticmethod
    def _get_name():
        return 'ParallelBatchNorm1d'


def fc_layer(in_features, out_features, activation=None, batchnorm=True):
    layers = [nn.Linear(in_features, out_features)]

    if activation is not None:
        layers += [activation]
    if batchnorm:
        layers += [ParallelBatchNorm1d(out_features)]

    return nn.Sequential(*layers)



class ResidualVideoAE(nn.Module):

    def __init__(self, input_shape, encoder_sizes, fc_sizes, *, temporal_strides,
                 decoder_sizes=None, color_channels=3, latent_activation=None):
        super().__init__()
        decoder_sizes = decoder_sizes if decoder_sizes is not None \
            else list(reversed(encoder_sizes))

        conv_dims = list(zip([color_channels, *encoder_sizes], encoder_sizes))
        temporal_strides = list(temporal_strides)
        assert len(temporal_strides) == len(conv_dims)

        self.conv_encoder = nn.Sequential(
            *[EncoderBlock(d_in, d_out, temporal_stride=ts)
              for (d_in, d_out), ts in zip(conv_dims, temporal_strides)])

        self.input_shape = (color_channels, *input_shape)
        with torch.no_grad():
            self.conv_encoder.eval()
            dummy_input = torch.Tensor(1, *self.input_shape)
            dummy_output = self.conv_encoder(dummy_input)
        _, c, t, h, w = dummy_output.shape
        self.intermediate_shape = (t, c, h, w)
        self.first_fc_size = (t, c * h * w)

        fc_dims = list(zip([self.first_fc_size[1], *fc_sizes], fc_sizes))
        if fc_dims:
            self.fc_encoder = nn.Sequential(
                *[fc_layer(d_in, d_out, activation=nn.LeakyReLU())
                  for d_in, d_out in fc_dims[:-1]],
                fc_layer(*fc_dims[-1], activation=latent_activation,
                         batchnorm=False))
        else:
            self.fc_encoder = nn.Sequential()

        self.fc_decoder = nn.Sequential(
            *[fc_layer(d_out, d_in, activation=nn.LeakyReLU())
              for d_in, d_out in reversed(fc_dims)])
        conv_dims = list(zip(decoder_sizes, [*decoder_sizes[1:], color_channels]))
        self.conv_decoder = nn.Sequential(
            *[DecoderBlock(d_in, d_out, temporal_stride=ts)
              for (d_in, d_out), ts in zip(conv_dims, reversed(temporal_strides))],
            nn.Sigmoid())

    def encode(self, x):
        """
        >>> model = ResidualVideoAE((16, 64, 64), [4], [], color_channels=4)
        >>> x = torch.randn(2, 4, 16, 64, 64)
        >>> y = model.encode(x)
        >>> tuple(y.shape)
        (2, 8, 4096)
        """
        y = self.conv_encoder(x)
        # Group together CWH indices, but keep time index separate
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(-1, *self.first_fc_size)
        y = self.fc_encoder(y)
        return y

    def decode(self, x):
        y = self.fc_decoder(x)
        y = y.view(-1, *self.intermediate_shape).permute(0, 2, 1, 3, 4)
        y = self.conv_decoder(y)
        return y

    def forward(self, x):
        """
        >>> model = ResidualVideoAE((16, 64, 64), [10], [5], color_channels=4)
        >>> x = torch.randn(2, 4, 16, 64, 64)
        >>> y = model(x)
        >>> tuple(y.shape)
        (2, 4, 16, 64, 64)
        """
        return self.decode(self.encode(x))
