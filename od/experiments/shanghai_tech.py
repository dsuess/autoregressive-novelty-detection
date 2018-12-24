from collections import namedtuple
from pathlib import Path
from time import time

import click
import matplotlib.pyplot as pl
import numpy as np
import nvvl
import od
import torch
import torchvision as tv
from od.datasets.videos import FrameMaskDataset
from od.utils import connected_compoents, logger, render_mpl_figure
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
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


class WrappedLabelledNVVL(WrappedNVVL):

    Result = namedtuple('Result', 'frames, y_gt, indices, video_name')

    def __iter__(self):
        for elem in self.loader:
            frames = elem[self.name]
            y_gt = torch.cat([x[None] for x, _, _ in elem['labels']])
            indices = torch.cat([x[None] for _, x, _ in elem['labels']])
            video_name = [x for _, _, x in elem['labels']]
            yield self.Result(frames, y_gt, indices, video_name)


def sample_from_nvvl(dataset, samples=10):
    indices = np.random.choice(len(dataset), size=samples, replace=False)
    return torch.cat([dataset[i]['input'].detach().to('cpu')[None] for i in indices])


def performance_plot(y_gt, y_pred):
    fig = pl.figure(0, figsize=(8, 4))
    x = np.arange(len(y_pred))
    pl.plot(x, y_pred)
    for start, end in connected_compoents(y_gt < 0.5):
        pl.axvspan(x[start], x[end], color='r', alpha=0.4)
    return fig


class ShanghaiTechExperiment(Experiment):

    time_steps = 16
    time_stride = 1
    frame_shape = (160, 256)
    batch_size = 64
    re_weight = .001

    def __init__(self, traindir, *args, **kwargs):
        height, width = self.frame_shape

        #  training_videos = list(sorted(map(str, Path(traindir).glob('*.mp4'))))
        training_videos = ['/home/daniel/data/shanghaitech/training/01.mp4']
        logger.info(f'Found {len(training_videos)} video files in training set.')

        testing_videos = [
            str(s) for s in Path('/home/daniel/data/shanghaitech/testing/videos/').glob('01_*.mp4')][:5]
        logger.info(f'Found {len(testing_videos)} video files in test set.')

        #  index_map = list(od.utils.interleave(
        #      range(self.time_steps),
        #      *[[-1] * self.time_steps] * (self.time_stride - 1)))
        index_map = np.arange(16)
        processing = {
            'input': nvvl.ProcessDesc(
                #  scale_width=width, scale_height=height,
                normalized=True, dimension_order='cfhw',
                index_map=list(index_map))}

        self.frame_mask_dataset = FrameMaskDataset(
            '/home/daniel/data/shanghaitech/testing/test_frame_mask',
            index_map=index_map, video_paths=testing_videos)
        # TODO Adjust for time_stride
        self.datasets = od.datasets.Split(
            nvvl.VideoDataset(
                training_videos, self.time_steps, device_id=0,
                processing=processing),
            nvvl.VideoDataset(
                testing_videos, self.time_steps, device_id=1,
                processing=processing, get_label=self.frame_mask_dataset.get_label))

        self.global_step = 0
        self.sample_images = sample_from_nvvl(self.datasets.train)

        super().__init__(*args, **kwargs)

    def get_loaders(self):
        return od.datasets.Split(
            WrappedNVVL(nvvl.VideoLoader(
                self.datasets.train, batch_size=self.batch_size, shuffle=True,
                buffer_length=3)),
            WrappedLabelledNVVL(nvvl.VideoLoader(
                self.datasets.test, batch_size=self.batch_size, shuffle=False,
                buffer_length=10)))

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

            if self.global_step % 2000 == 0:
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

    def eval_test_dataset(self, loader):
        frames_dict = self.frame_mask_dataset.nr_frames
        groundtruth = {stem: torch.zeros(nr_frames)
                       for stem, nr_frames in frames_dict.items()}
        prediction = {stem: torch.zeros(nr_frames)
                      for stem, nr_frames in frames_dict.items()}
        counts = {stem: torch.zeros(nr_frames)
                  for stem, nr_frames in frames_dict.items()}

        for result in loader:
            y_pred = self.model.module.predict(result.frames.to(self.device)).to('cpu')

            for name, idx, g, p in zip(result.video_name, result.indices, result.y_gt, y_pred):
                stem = Path(name).stem
                groundtruth[stem][idx] += g
                prediction[stem][idx] += p
                counts[stem][idx] += 1

        for stem in groundtruth:
            sel = counts[stem].numpy() > 0
            groundtruth[stem] = (groundtruth[stem] / counts[stem]).numpy()[sel]
            prediction[stem] = (prediction[stem] / counts[stem]).numpy()[sel]
        return groundtruth, prediction

    def eval_epoch(self, epoch, summary_writer):
        super().eval_epoch(epoch, summary_writer)

        with torch.no_grad():
            sample_images = self.make_example_images(self.sample_images)
            for i, sample in enumerate(sample_images):
                summary_writer.add_image(f'train_images_{i}', sample, epoch)

            roc_scores = []
            y_gts, y_preds = self.eval_test_dataset(self.loaders.test)

            for key in y_gts:
                y_gt, y_pred = y_gts[key], y_preds[key]
                roc_scores.append(metrics.roc_auc_score(y_gt, y_pred))
                fig = performance_plot(y_gt, y_pred)
                fig = render_mpl_figure(fig)
                summary_writer.add_image(f'test_performance_{key}', fig, epoch)

            roc_score = np.mean(roc_scores)
            summary_writer.add_scalar(f'metrics/avg_roc_auc', roc_score, epoch)


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

    def run(self, keep_every_ckpt=True):
        summary_writer = SummaryWriter(self.logdir)
        restored_epoch = self.restore_latest(self.checkpoint_dir)
        epochs = tqdm(range(restored_epoch + 1, self.epochs + 1),
                      total=self.epochs, initial=restored_epoch)
        epochs.refresh()
        for epoch in epochs:
            epochs.update(epoch)
            self.train_epoch(epoch, summary_writer)

            if keep_every_ckpt:
                filename = self.checkpoint_dir / f'checkpoint_{epoch:04d}.pt'
            else:
                filename = self.checkpoint_dir / 'checkpoint.pt'
            self.save(filename, epoch)
