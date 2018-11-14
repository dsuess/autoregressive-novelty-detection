import os
from collections import namedtuple
from pathlib import Path

import click
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm, trange

from .autoregress import AutoregresionModule, AutoregressiveLoss
from .datasets import UnsupervisedMNIST
from .encoders import ResidualAE
from .utils import logger

WRITE_DIRECTORY = click.Path(file_okay=False, resolve_path=True, writable=True)


def sample_examples(dataset, size):
    sel = np.random.choice(len(dataset), size=min(size, len(dataset)), replace=False)
    return torch.cat([dataset[i][None] for i in sel], 0)


class Experiment:

    Datasets = namedtuple('Datasets', 'train, test')

    def __init__(self, datasets, conv_channels, fc_channels, mfc_channels, batch_size, log_dir):
        self.datasets = self.Datasets(*datasets)
        self.loaders = self.Datasets(
            DataLoader(self.datasets.train, batch_size=batch_size, shuffle=True,
                       pin_memory=True),
            DataLoader(self.datasets.test, batch_size=batch_size))
        self.sample_images = self.Datasets(
            sample_examples(self.datasets.train, size=50),
            sample_examples(self.datasets.test, size=50))

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using pytorch device={self.device}')

        color_channels, *input_shape = self.datashape
        model = AutoregressiveLoss(
            ResidualAE(input_shape, conv_channels, fc_channels, color_channels=color_channels),
            AutoregresionModule(fc_channels[-1], mfc_channels))
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters())

        Path(log_dir).mkdir(exist_ok=True)
        self.log_dir = log_dir
        self.checkpoint_dir = Path(log_dir) / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

    @property
    def datashape(self):
        return self.datasets.train[0].shape

    def restore(self, ckpt_path):
        logger.info(f'Restoring model from {ckpt_path}')
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        logger.info(f'Successfully restored at epoch={epoch}')
        return epoch

    def restore_latest(self, ckpt_dir):
        files = ckpt_dir.glob('*.pt')

        try:
            latest = max(files, key=os.path.getctime)
        except ValueError:
            logger.info(f'No checkpoints found in {ckpt_dir}')
            return 0
        return self.restore(latest)

    def run(self):
        summary_writer = SummaryWriter(self.log_dir)
        restored_epoch = self.restore_latest(self.checkpoint_dir)
        # Batch size == 1 fails due to batch norm in training mode
        if restored_epoch <= 0:
            dummy_input = torch.Tensor(3, *self.datashape).to(self.device)
            summary_writer.add_graph(self.model, dummy_input)

        for epoch in trange(restored_epoch + 1, 11):

            loss_summary = {
                'total': torch.zeros(1, dtype=torch.float, device=self.device),
                'reconstruction': torch.zeros(1, dtype=torch.float, device=self.device),
                'autoregressive': torch.zeros(1, dtype=torch.float, device=self.device)}

            for x in tqdm(self.loaders.train):
                x = x.to(self.device)
                losses = self.model(x)
                loss = losses.reconstruction + self.model.autoreg.bins * losses.autoregressive

                loss_summary['total'] += loss
                loss_summary['reconstruction'] += losses.reconstruction
                loss_summary['autoregressive'] += losses.autoregressive

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            for name, value in loss_summary.items():
                summary_writer.add_scalar(
                    name, value / len(self.datasets.train), epoch)

            state = {
                'epoch': epoch, 'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}
            filename = self.checkpoint_dir / f'checkpoint_{epoch:04d}.pt'
            logger.info(f'Saving checkpoint to {filename}')
            torch.save(state, filename)

            #  model.eval()
            #  for n, img in enumerate(test_images):
                #  img_pred = model.autoencoder(img[None].to(device))[0]
                #  merged = make_grid([img.cpu(), img_pred.cpu()])
                #  summary_writer.add_image(f'image_{n}', merged, epoch)
            #  model.train()


@click.group()
def experiments():
    pass


@experiments.command()
@click.option('--log-dir', required=True, type=WRITE_DIRECTORY)
def mnist(log_dir):
    all_digits = set(range(10))
    test_digits = {3, 5, 8}
    train_digits = all_digits.difference(test_digits)
    ds_train = UnsupervisedMNIST(
        exclude_digits=test_digits, transform=transforms.ToTensor())
    ds_test = UnsupervisedMNIST(
        exclude_digits=train_digits, transform=transforms.ToTensor())

    Experiment(
        datasets=(ds_train, ds_test),
        conv_channels=[32, 64],
        fc_channels=[64],
        mfc_channels=[32, 32 ,32 ,32, 100],
        batch_size=64,
        log_dir=log_dir).run()
