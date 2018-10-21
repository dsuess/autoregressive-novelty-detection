import abc

import torch
from torch import nn


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

    def __init__(self, in_channels, out_channels, raw_output=False):
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


class ResidualAE(nn.Module):

    def __init__(self, conv_filters, color_channels=3):
        super().__init__()

        dimensions = list(zip([color_channels, *conv_filters], conv_filters))
        self.encoder = nn.Sequential(
            *[EncoderBlock(d_in, d_out) for d_in, d_out in dimensions])
        self.decoder = nn.Sequential(
            *[DecoderBlock(d_in, d_out) for d_out, d_in in reversed(dimensions)],
            nn.Sigmoid())

    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        return y

