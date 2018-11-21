import os
from collections import namedtuple
from multiprocessing import cpu_count
from pathlib import Path

import click
import matplotlib.pyplot as pl
import numpy as np
import od
import torch
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

    def __init__(self, datasets, conv_channels, fc_channels, mfc_channels,
                 batch_size, nr_epochs, logdir, settings, ar_weight):
        self.datasets = self.Datasets(*datasets)
        logger.info(f'Got {len(self.datasets.train)} training examples and '
                    f'{len(self.datasets.test)} test examples')
        self.loaders = self.Datasets(
            DataLoader(self.datasets.train, batch_size=batch_size,
                       shuffle=True, num_workers=2 * cpu_count()),
            DataLoader(self.datasets.test, batch_size=batch_size,
                       num_workers=2 * cpu_count()))
        self.sample_images = self.Datasets(
            sample_examples(self.datasets.train, size=10),
            sample_examples(self.datasets.test, size=10))

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using pytorch device={self.device}')

        color_channels, *input_shape = self.datashape
        encoder = od.ResidualAE(
            input_shape, conv_channels, fc_channels,
            color_channels=color_channels, latent_activation=nn.Sigmoid())
        regressor = od.AutoregresionModule(
            fc_channels[-1], mfc_channels,
            batchnorm=settings.get('autoregress_batchnorm', True))
        model = od.AutoregressiveLoss(encoder, regressor)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.ar_weight = ar_weight

        Path(logdir).mkdir(exist_ok=True)
        self.nr_epochs = nr_epochs
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
            losses = self.model.Result(
                losses.reconstruction.mean(), losses.autoregressive.mean())
            loss = losses.reconstruction + self.ar_weight * losses.autoregressive

            loss_summary['loss/total'] += loss
            loss_summary['loss/reconstruction'] += losses.reconstruction
            loss_summary['loss/autoregressive'] += losses.autoregressive

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        nr_examples = len(self.datasets.train)
        for name, value in loss_summary.items():
            summary_writer.add_scalar(name, value / nr_examples, epoch)

    def make_example_images(self, sample_images):
        imgs_pred = self.model.encoder(sample_images.to(self.device))
        img_pairs = zip(sample_images, imgs_pred.cpu())
        img_merged = (make_grid(list(pair), nrow=1) for pair in img_pairs)
        all_images = make_grid(list(img_merged), nrow=len(sample_images))
        return all_images


    def eval_epoch(self, epoch, summary_writer):
        self.model.eval()
        with torch.no_grad():
            examples = self.make_example_images(self.sample_images.train)
            summary_writer.add_image('train_images', examples, epoch)
            examples = self.make_example_images(self.sample_images.test)
            summary_writer.add_image('test_images', examples, epoch)

            train_losses = np.concatenate(
                [self.model.predict(x.to(self.device)).to('cpu').numpy()
                 for x, in self.loaders.train])
            test_losses = np.concatenate(
                [self.model.predict(x.to(self.device)).to('cpu').numpy()
                 for x, in self.loaders.test])

            fig = pl.figure(0, figsize=(10, 10))
            pl.hist(train_losses, bins=100, density=True)
            pl.hist(test_losses, alpha=0.5, bins=100, density=True)
            rendered_fig = od.utils.render_mpl_figure()
            summary_writer.add_image('loss_histogram', rendered_fig, epoch)

            overlap = od.utils.sample_distribution_overlap(train_losses, test_losses)
            summary_writer.add_scalar('histogram_overlap', overlap, epoch)


    def run(self):
        summary_writer = SummaryWriter(self.logdir)
        restored_epoch = self.restore_latest(self.checkpoint_dir)
        self.eval_epoch(0, summary_writer)
        epochs = tqdm(range(restored_epoch + 1, self.nr_epochs + 1),
                      total=self.nr_epochs, initial=restored_epoch)
        epochs.refresh()
        for epoch in epochs:
            epochs.update(epoch)
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
    ds_train = od.UnsupervisedMNIST(exclude_digits=test_digits)
    ds_test = od.UnsupervisedMNIST(exclude_digits=train_digits)

    Experiment(
        datasets=(ds_train, ds_test),
        conv_channels=[32, 64],
        fc_channels=[64],
        mfc_channels=[32, 32 ,32 ,32, 100],
        batch_size=64,
        nr_epochs=50,
        logdir=logdir,
        settings={'autoregress_batchnorm': False},
        ar_weight=10).run()
