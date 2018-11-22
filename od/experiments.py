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
from od.datasets import Split
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange

WRITE_DIRECTORY = click.Path(file_okay=False, resolve_path=True, writable=True)


def sample_examples(dataset, size, should_be_known):
    loader = DataLoader(dataset, shuffle=True, batch_size=1)
    result = []
    for x, y in loader:
        if y == should_be_known:
            result.append(x)
            if len(result) >= size:
                break
    return torch.cat(result, dim=0)


class Experiment:

    def __init__(self, datasets, conv_channels, fc_channels, mfc_channels,
                 batch_size, nr_epochs, logdir, settings, ar_weight,
                 encoder_decay):
        pl.style.use('ggplot')
        self.datasets = Split(*datasets)
        self.loaders = Split(
            DataLoader(self.datasets.train, batch_size=batch_size,
                       shuffle=True, num_workers=2 * cpu_count()),
            DataLoader(self.datasets.test, batch_size=batch_size,
                       num_workers=2 * cpu_count()))
        self.sample_images = Split(
            sample_examples(self.datasets.train, size=10, should_be_known=True),
            sample_examples(self.datasets.test, size=10, should_be_known=False))

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
        print(model)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.ar_weight = ar_weight
        self.encoder_decay = encoder_decay

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

        for x, _ in tqdm(self.loaders.train):
            x = x.to(self.device)
            losses = self.model(x)
            losses = self.model.Result(
                losses.reconstruction.mean(), losses.autoregressive.mean())
            loss = losses.reconstruction + self.ar_weight * losses.autoregressive

            nr_examples = x.size(0)
            loss_summary['loss/total'] += loss * nr_examples
            loss_summary['loss/reconstruction'] += losses.reconstruction * nr_examples
            loss_summary['loss/autoregressive'] += losses.autoregressive * nr_examples

            self.optimizer.zero_grad()
            loss.backward()

            if self.encoder_decay is not None:
                for param in self.model.encoder.parameters():
                    param.grad *= self.encoder_decay**epoch
            self.optimizer.step()

        nr_examples = len(self.datasets.train)
        for name, value in loss_summary.items():
            summary_writer.add_scalar(name, value / nr_examples, epoch)

        if self.encoder_decay is not None:
            summary_writer.add_scalar(
                'loss/encoder_decay', self.encoder_decay**epoch, epoch)

    def make_example_images(self, sample_images):
        imgs_pred = self.model.encoder(sample_images.to(self.device))
        img_pairs = zip(sample_images, imgs_pred.cpu())
        img_merged = (make_grid(list(pair), nrow=1) for pair in img_pairs)
        all_images = make_grid(list(img_merged), nrow=len(sample_images))
        return all_images

    def _compute_eval_losses(self):
        train_losses = np.concatenate(
            [self.model.predict(x.to(self.device)).to('cpu').numpy()
             for x, _ in self.loaders.train])
        test_losses = np.concatenate(
            [self.model.predict(x.to(self.device)).to('cpu').numpy()
                for x, _ in self.loaders.test])
        is_known = np.concatenate([y for _, y in self.loaders.test]).astype(bool)

        known_lossses = np.concatenate([train_losses, test_losses[is_known]])
        unknown_losses = test_losses[~is_known]
        return known_lossses, unknown_losses

    def eval_epoch(self, epoch, summary_writer):
        self.model.eval()
        with torch.no_grad():
            examples = self.make_example_images(self.sample_images.train)
            summary_writer.add_image('train_images', examples, epoch)
            examples = self.make_example_images(self.sample_images.test)
            summary_writer.add_image('test_images', examples, epoch)

            known_losses, unknown_losses = self._compute_eval_losses()
            fig = pl.figure(0, figsize=(6, 6))
            pl.hist(known_losses, bins=100, density=True, label='known')
            pl.hist(unknown_losses, alpha=0.5, bins=100, density=True,
                    label='unknown')
            pl.title('Loss Histogram')
            pl.xlabel('Autoreg. Loss')
            pl.legend()
            rendered_fig = od.utils.render_mpl_figure()
            summary_writer.add_image('loss_histogram', rendered_fig, epoch)

            overlap = od.utils.sample_distribution_overlap(
                known_losses, unknown_losses)
            summary_writer.add_scalar('metrics/histogram_overlap', overlap, epoch)
            summary_writer.add_scalar('metrics/train_loss', np.mean(known_losses), epoch)
            summary_writer.add_scalar('metrics/test_loss', np.mean(unknown_losses), epoch)

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
@click.option('--ar-weight', default=1.0, type=float)
@click.option('--batchnorm', default=True, type=bool)
@click.option('--encoder-decay', type=float)
def mnist(logdir, ar_weight, batchnorm, encoder_decay):
    Experiment(
        datasets=od.mnist_novelty_dataset(novel_digits={3, 5, 8}),
        conv_channels=[32, 64],
        fc_channels=[64],
        mfc_channels=[32, 32 ,32 ,32, 100],
        batch_size=64,
        nr_epochs=50,
        logdir=logdir,
        settings={'autoregress_batchnorm': batchnorm},
        ar_weight=ar_weight,
        encoder_decay=encoder_decay).run()
