import abc
import json
import os
import tempfile
from collections import namedtuple
from multiprocessing import cpu_count
from pathlib import Path
from time import time

import click
import matplotlib.pyplot as pl
import numpy as np
import nvvl
import od
import torch
import torchvision as tv
from od.datasets import Split
from od.utils import logger, mse_loss
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

WRITE_DIRECTORY = click.Path(file_okay=False, resolve_path=True, writable=True)


def sample_examples(dataset, size, should_be_known):
    loader = DataLoader(dataset, shuffle=True, batch_size=1)
    result = []
    for elem in loader:
        if len(elem) == 1:
            result.append(elem)
        elif elem[1] == should_be_known:
            result.append(elem[0])

        if len(result) >= size:
            break
    return torch.cat(result, dim=0)


class Experiment:

    TestLosses = namedtuple('TestLosses', 'known, unknown, test, test_known')

    re_weight = 1.0
    ar_weight = 1.0
    learning_rate = 1e-3

    def __init__(self, epochs, logdir):
        pl.style.use('ggplot')
        self.loaders = self.get_loaders()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using pytorch device={self.device}')

        self.model = self.get_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate * self.batch_size)

        Path(logdir).mkdir(exist_ok=True)
        self.epochs = epochs
        self.logdir = logdir
        self.checkpoint_dir = Path(logdir) / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

    @abc.abstractmethod
    def get_loaders(self):
        pass

    @abc.abstractstaticmethod
    def get_model():
        pass

    def restore(self, ckpt_path):
        logger.info(f'Restoring model from {ckpt_path}')
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        logger.info(f'Successfully restored at epoch={epoch}')
        return epoch

    def save(self, ckpt_path, epoch):
        state = {
            'epoch': epoch, 'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()}
        logger.info(f'Saving checkpoint to {ckpt_path}')
        torch.save(state, ckpt_path)

    def restore_latest(self, ckpt_dir):
        files = ckpt_dir.glob('*.pt')

        try:
            latest = max(files, key=os.path.getctime)
        except ValueError:
            logger.info(f'No checkpoints found in {ckpt_dir}')
            return 0
        return self.restore(latest)

    def train_step(self, x):
        x = x.to(self.device)
        losses = self.model(x)
        losses = {key: val.mean() for key, val in losses.items()}
        total_loss = self.re_weight * losses['reconstruction'] \
            + self.ar_weight * losses['autoregressive']

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss, losses

    def train_epoch(self, epoch, summary_writer):
        self.model.train()

        loss_summary = {
            'loss/total': torch.zeros(1, dtype=torch.float, device=self.device),
            'loss/reconstruction': torch.zeros(1, dtype=torch.float, device=self.device),
            'loss/autoregressive': torch.zeros(1, dtype=torch.float, device=self.device)}

        for x in tqdm(self.loaders.train):
            loss, losses = self.train_step(x)

            nr_examples = x.size(0)
            loss_summary['loss/total'] += loss * nr_examples
            loss_summary['loss/reconstruction'] += losses['reconstruction'] * nr_examples
            loss_summary['loss/autoregressive'] += losses['autoregressive'] * nr_examples

        nr_examples = len(self.datasets.train)
        for name, value in loss_summary.items():
            summary_writer.add_scalar(name, value / nr_examples, epoch)

    def eval_epoch(self, epoch, summary_writer):
        self.model.eval()

    def run(self, keep_every_ckpt=True):
        summary_writer = SummaryWriter(self.logdir)
        restored_epoch = self.restore_latest(self.checkpoint_dir)
        self.eval_epoch(0, summary_writer)
        epochs = tqdm(range(restored_epoch + 1, self.epochs + 1),
                      total=self.epochs, initial=restored_epoch)
        epochs.refresh()
        for epoch in epochs:
            epochs.update(epoch)
            self.train_epoch(epoch, summary_writer)
            self.eval_epoch(epoch, summary_writer)

            if keep_every_ckpt:
                filename = self.checkpoint_dir / f'checkpoint_{epoch:04d}.pt'
            else:
                filename = self.checkpoint_dir / 'checkpoint.pt'
            self.save(filename, epoch)



class ImageClassifierExperiment(Experiment):

    def __init__(self, datasets, *args, **kwargs):
        self.datasets = Split(*datasets)
        super().__init__(*args, **kwargs)
        self.sample_images = Split(
            sample_examples(self.datasets.train, size=10, should_be_known=True),
            sample_examples(self.datasets.test, size=10, should_be_known=False))

    def get_loaders(self):
        return Split(
            DataLoader(self.datasets.train, batch_size=self.batch_size,
                       shuffle=True, num_workers=cpu_count()),
            DataLoader(self.datasets.test, batch_size=self.batch_size,
                       num_workers=cpu_count()))

    def make_example_images(self, sample_images):
        imgs_pred = self.model.encoder(sample_images.to(self.device))
        img_pairs = zip(sample_images, imgs_pred.cpu())
        img_merged = (make_grid(list(pair), nrow=1) for pair in img_pairs)
        all_images = make_grid(list(img_merged), nrow=len(sample_images))
        return all_images

    def compute_eval_losses(self):
        train_losses = np.concatenate(
            [self.model.predict(x.to(self.device)).to('cpu').numpy()
             for x in self.loaders.train])
        data = [(self.model.predict(x.to(self.device)).to('cpu').numpy(), y.numpy())
                for x, y in self.loaders.test]
        test_losses = np.concatenate([x for x, _ in data])
        is_known = np.concatenate([y for _, y in data]).astype(bool)

        known_lossses = np.concatenate([train_losses, test_losses[is_known]])
        unknown_losses = test_losses[~is_known]
        return self.TestLosses(known_lossses, unknown_losses, test_losses, is_known)

    def eval_epoch(self, epoch, summary_writer):
        super().eval_epoch(epoch, summary_writer)

        with torch.no_grad():
            examples = self.make_example_images(self.sample_images.train)
            summary_writer.add_image('train_images', examples, epoch)
            examples = self.make_example_images(self.sample_images.test)
            summary_writer.add_image('test_images', examples, epoch)

            losses = self.compute_eval_losses()
            pl.figure(0, figsize=(6, 6))
            pl.hist(losses.known, bins=100, density=True, label='known')
            pl.hist(losses.unknown, alpha=0.5, bins=100, density=True,
                    label='unknown')
            pl.title('Loss Histogram')
            pl.xlabel('Autoreg. Loss')
            pl.legend()
            rendered_fig = od.utils.render_mpl_figure()
            summary_writer.add_image('loss_histogram', rendered_fig, epoch)

            overlap = od.utils.sample_distribution_overlap(
                losses.known, losses.unknown)
            summary_writer.add_scalar('metrics/histogram_overlap', overlap, epoch)
            summary_writer.add_scalar('metrics/train_loss', np.mean(losses.known), epoch)
            summary_writer.add_scalar('metrics/test_loss', np.mean(losses.unknown), epoch)

            # "-" because autoreg. losses approximates neg. log. likelihood
            roc_score = metrics.roc_auc_score(losses.test_known, -losses.test)
            summary_writer.add_scalar('metrics/roc_auc', roc_score, epoch)



@click.group()
def experiments():
    pass


class MNISTExperiment(ImageClassifierExperiment):

    batch_size = 64

    @staticmethod
    def get_model():
        encoder = od.ResidualAE(
            input_shape=(28, 28),
            encoder_sizes=[32, 64],
            fc_sizes=[64],
            color_channels=1,
            latent_activation=nn.Sigmoid())

        regressor = od.AutoregresionModule(
            dim=64,
            layer_sizes=[32, 32, 32, 32, 100])

        return od.AutoregressiveLoss(encoder, regressor)


@experiments.command(name='mnist')
@click.option('--logdir', required=True, type=WRITE_DIRECTORY)
@click.option('--download-dir', required=False, type=WRITE_DIRECTORY)
@click.option('--epochs', default=50, type=int)
@click.option('--batch', is_flag=True)
def mnist(logdir, download_dir, epochs, batch):
    transforms = [tv.transforms.RandomAffine(degrees=20, shear=20)]
    if download_dir is None:
        download_dir = Path(tempfile.gettempdir()) / 'mnist'

    if batch:
        logdir = Path(logdir)
        logdir.mkdir(exist_ok=True)
        result = dict()

        for i in range(10):
            datasets = od.datasets.MNIST.load_split(
                download_dir, {i}, download=True, transforms=transforms)
            experiment = MNISTExperiment(
                datasets=datasets, logdir=logdir / f'only_{i}', epochs=epochs)
            experiment.run(keep_every_ckpt=False)

            losses = experiment.compute_eval_losses()
            roc_score = metrics.roc_auc_score(losses.test_known, -losses.test)
            result[i] = roc_score

        with open(logdir / 'result.json', 'w') as buf:
            json.dump(result, buf)

    else:
        datasets = od.datasets.MNIST.load_split(
            download_dir, {1, 2, 3}, download=True, transforms=transforms)
        MNISTExperiment(datasets=datasets, logdir=logdir, epochs=epochs).run()


class CIFAR10Experiment(ImageClassifierExperiment):

    batch_size = 64

    @staticmethod
    def get_model():
        encoder = od.ResidualAE(
            input_shape=(32, 32),
            encoder_sizes=[64, 128, 256],
            fc_sizes=[256, 64],
            color_channels=3,
            latent_activation=nn.Sigmoid())

        regressor = od.AutoregresionModule(
            dim=64,
            layer_sizes=[32, 32, 32, 32, 100])

        return od.AutoregressiveLoss(encoder, regressor)

@experiments.command(name='cifar10')
@click.option('--logdir', required=True, type=WRITE_DIRECTORY)
@click.option('--download-dir', required=False, type=WRITE_DIRECTORY)
@click.option('--epochs', default=50, type=int)
@click.option('--batch', is_flag=True)
def cifar10(logdir, download_dir, epochs, batch):
    if download_dir is None:
        download_dir = Path(tempfile.gettempdir()) / 'cifar10'

    if batch:
        logdir = Path(logdir)
        logdir.mkdir(exist_ok=True)
        result = dict()

        for i in range(10):
            datasets = od.datasets.CIFAR10.load_split(
                download_dir, {i}, download=True)
            experiment = CIFAR10Experiment(
                datasets=datasets, logdir=logdir / f'only_{i}', epochs=epochs)
            experiment.run(keep_every_ckpt=False)

            losses = experiment.compute_eval_losses()
            roc_score = metrics.roc_auc_score(losses.test_known, -losses.test)
            result[i] = roc_score

        with open(logdir / 'result.json', 'w') as buf:
            json.dump(result, buf)

    else:
        datasets = od.datasets.CIFAR10.load_split(
            download_dir, {1, 2, 3}, download=True)
        CIFAR10Experiment(datasets=datasets, logdir=logdir, epochs=epochs).run()


# FIXME Clean up train code so we don't need this anymore
class WrappedNVVL:

    def __init__(self, loader, name='input'):
        self.loader = loader
        self.name = name

    def __iter__(self):
        return (elem[self.name] for elem in self.loader)


class ShanghaiTechExperiment(Experiment):

    time_steps = 16
    time_stride = 1
    frame_shape = (160, 256)
    batch_size = 16
    re_weight = 0.0001

    def __init__(self, traindir, *args, **kwargs):
        height, width = self.frame_shape
        video_files = list(sorted(map(str, Path(traindir).glob('*.mp4'))))

        video_files = ['/home/daniel/data/shanghaitech/training/01.mp4']
        logger.info(f'Found {len(video_files)} video files in training set.')
        index_map = list(od.utils.interleave(
            range(self.time_steps),
            *[[-1] * self.time_steps] * (self.time_stride - 1)))
        processing = {
            'input': nvvl.ProcessDesc(
                #  scale_width=width, scale_height=height,
                normalized=True, dimension_order='cfhw', index_map=index_map)}
        trainset = nvvl.VideoDataset(
            video_files, self.time_stride * self.time_steps, device_id=0,
            processing=processing)

        indices = np.random.choice(len(trainset), size=10, replace=False)
        self.sample_images = torch.cat(
            [trainset[i]['input'].detach().to('cpu')[None] for i in indices])
        self.datasets = Split(trainset, None)
        self.global_step = 0

        super().__init__(*args, **kwargs)

    def get_loaders(self):
        return Split(
            WrappedNVVL(nvvl.VideoLoader(
                self.datasets.train, batch_size=self.batch_size, shuffle=True,
                buffer_length=3)),
            None)

    @classmethod
    def get_model(cls, parallel=True):
        encoder = od.ResidualVideoAE(
            input_shape=(cls.time_steps, *cls.frame_shape),
            encoder_sizes=[8, 16, 32, 64, 64],
            decoder_sizes=[64, 32, 16, 8, 8],
            temporal_strides=[2, 2, 1, 1, 1],
            fc_sizes=[512, 64],
            color_channels=3,
            latent_activation=nn.Sigmoid())

        regressor = od.AutoregresionModule(
            dim=64,
            layer_sizes=[32, 32, 32, 32, 100],
            layer=od.AutoregressiveConvLayer)

        model = od.AutoregressiveVideoLoss(encoder, regressor)
        if parallel:
            model = nn.DataParallel(model)
        return model

    def train_epoch(self, epoch, summary_writer):
        self.model.train()

        for x in tqdm(self.loaders.train):
            start_time = time()
            loss, losses = self.train_step(x)
            runtime = time() - start_time

            summary_writer.add_scalar(
                'loss/it_per_s', 1 / runtime, self.global_step)
            summary_writer.add_scalar(
                'loss/total', loss, self.global_step)
            summary_writer.add_scalar(
                'loss/reconstruction', losses['reconstruction'], self.global_step)
            summary_writer.add_scalar(
                'loss/autoregressive', losses['autoregressive'], self.global_step)

            if self.global_step % 100 == 0:
                self.eval_epoch(self.global_step, summary_writer)
                self.model.train()

            self.global_step += 1

    def make_example_images(self, sample_images):
        reconstructions = self.model.module.encoder(
            sample_images.to(self.device)).to('cpu')

        for original, recons in zip(sample_images, reconstructions):
            original = make_grid(original.transpose(0, 1), nrow=original.size(1))
            recons = make_grid(recons.transpose(0, 1), nrow=original.size(1))
            yield make_grid([original, recons], nrow=1)

    def eval_epoch(self, epoch, summary_writer):
        super().eval_epoch(epoch, summary_writer)

        with torch.no_grad():
            sample_images = self.make_example_images(self.sample_images)
            for i, sample in enumerate(sample_images):
                summary_writer.add_image(f'train_images_{i}', sample, epoch)

    def save(self, ckpt_path, epoch):
        state = {
            'epoch': epoch, 'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step}
        logger.info(f'Saving checkpoint to {ckpt_path}')
        torch.save(state, ckpt_path)

    def restore(self, ckpt_path):
        logger.info(f'Restoring model from {ckpt_path}')
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        logger.info(f'Successfully restored at epoch={epoch}')
        return epoch


@experiments.command('shanghai-tech')
@click.option('--logdir', required=True, type=WRITE_DIRECTORY)
@click.option('--epochs', default=50, type=int)
def shanghai_tech(logdir, epochs):
    ShanghaiTechExperiment(
        traindir='/home/daniel/data/shanghaitech/training/h264',
        logdir=logdir, epochs=epochs).run()
