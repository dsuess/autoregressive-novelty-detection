import os
from collections import namedtuple
from multiprocessing import cpu_count
from pathlib import Path

import click
import numpy as np
import torch
from od.autoregress import AutoregresionModule, AutoregressiveLoss
from od.datasets import UnsupervisedMNIST
from od.encoders import ResidualAE
from od.utils import logger
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange

WRITE_DIRECTORY = click.Path(file_okay=False, resolve_path=True, writable=True)


def sample_examples(dataset, size):
    sel = np.random.choice(len(dataset), size=min(size, len(dataset)), replace=False)
    return torch.cat([dataset[i][0][None] for i in sel], 0)


class Experiment:

    Datasets = namedtuple('Datasets', 'train, test')

    def __init__(self, datasets, conv_channels, fc_channels, mfc_channels, batch_size, logdir):
        self.datasets = self.Datasets(*datasets)
        self.loaders = self.Datasets(
            DataLoader(self.datasets.train, batch_size=batch_size,
                       shuffle=True, num_workers=2 * cpu_count()),
            DataLoader(self.datasets.test, batch_size=batch_size,
                       num_workers=2 * cpu_count()))
        self.sample_images = self.Datasets(
            sample_examples(self.datasets.train, size=50),
            sample_examples(self.datasets.test, size=50))

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using pytorch device={self.device}')

        color_channels, *input_shape = self.datashape
        encoder = ResidualAE(input_shape, conv_channels, fc_channels,
                             color_channels=color_channels,
                             latent_activation=nn.Sigmoid())
        regressor = AutoregresionModule(fc_channels[-1], mfc_channels)
        model = AutoregressiveLoss(encoder, regressor)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters())

        Path(logdir).mkdir(exist_ok=True)
        self.logdir = logdir
        self.checkpoint_dir = Path(logdir) / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

    @property
    def datashape(self):
        return self.datasets.train[0][0].shape

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

    def train_epoch(self, epoch, summary_writer):
        self.model.train()
        loss_summary = {
            'loss/total': torch.zeros(1, dtype=torch.float, device=self.device),
            'loss/reconstruction': torch.zeros(1, dtype=torch.float, device=self.device),
            'loss/autoregressive': torch.zeros(1, dtype=torch.float, device=self.device)}

        for x, in tqdm(self.loaders.train):
            x = x.to(self.device)
            losses = self.model(x)
            loss = losses.reconstruction + losses.autoregressive

            loss_summary['loss/total'] += loss
            loss_summary['loss/reconstruction'] += losses.reconstruction
            loss_summary['loss/autoregressive'] += losses.autoregressive

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        nr_examples = len(self.datasets.train)
        for name, value in loss_summary.items():
            summary_writer.add_scalar(name, value / nr_examples, epoch)

    def eval_epoch(self, epoch, summary_writer):
        self.model.eval()
        for n, img in enumerate(self.sample_images.train):
            img_pred, = self.model.encoder(img[None].to(self.device))
            merged = make_grid([img, img_pred.cpu()])
            summary_writer.add_image(f'train_{n}', merged, epoch)

        for n, img in enumerate(self.sample_images.test):
            img_pred, = self.model.encoder(img[None].to(self.device))
            merged = make_grid([img, img_pred.cpu()])
            summary_writer.add_image(f'test_{n}', merged, epoch)

    def run(self):
        summary_writer = SummaryWriter(self.logdir)
        restored_epoch = self.restore_latest(self.checkpoint_dir)
        if restored_epoch <= 0:
            # Batch size == 1 fails due to batch norm in training mode
            dummy_input = torch.Tensor(3, *self.datashape).to(self.device)
            summary_writer.add_graph(self.model, dummy_input)

        self.eval_epoch(0, summary_writer)
        for epoch in trange(restored_epoch + 1, 11):
            self.train_epoch(epoch, summary_writer)
            self.eval_epoch(epoch, summary_writer)

            state = {
                'epoch': epoch, 'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}
            filename = self.checkpoint_dir / f'checkpoint_{epoch:04d}.pt'
            logger.info(f'Saving checkpoint to {filename}')
            torch.save(state, filename)


@click.group()
def experiments():
    pass


@experiments.command()
@click.option('--logdir', required=True, type=WRITE_DIRECTORY)
def mnist(logdir):
    all_digits = set(range(10))
    test_digits = {3, 5, 8}
    train_digits = all_digits.difference(test_digits)
    ds_train = UnsupervisedMNIST(exclude_digits=test_digits)
    ds_test = UnsupervisedMNIST(exclude_digits=train_digits)

    Experiment(
        datasets=(ds_train, ds_test),
        conv_channels=[32, 64],
        fc_channels=[64],
        mfc_channels=[32, 32 ,32 ,32, 100],
        batch_size=64,
        logdir=logdir).run()
