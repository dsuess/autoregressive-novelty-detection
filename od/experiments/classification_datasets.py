import json
import tempfile
from multiprocessing import cpu_count
from pathlib import Path

import click
import matplotlib.pyplot as pl
import numpy as np
import od
import torch
import torchvision as tv
from od.datasets import Split
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from .base import Experiment


__all__ = ['MNISTExperiment', 'CIFAR10Experiment']


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
