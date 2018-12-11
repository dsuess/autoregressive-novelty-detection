from pathlib import Path
from time import time

import click
import matplotlib.pyplot as pl
import numpy as np
import nvvl
import od
import torch
from od.datasets import Split
from od.utils import logger
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm
from .base import Experiment


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
