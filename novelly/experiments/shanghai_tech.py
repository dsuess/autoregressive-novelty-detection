from collections import namedtuple
from pathlib import Path
from time import time

import click
import ignite
import numpy as np
import torch
import torchvision as tv
from ignite._utils import convert_tensor
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

import matplotlib.pyplot as pl
import novelly
from novelly.datasets.videos import FrameMaskDataset
from novelly.utils import connected_compoents, logger, render_mpl_figure

from .base import Experiment

try:
    import nvvl
except ModuleNotFoundError:
    import warnings
    warnings.warn('Could not import NVVL')


def sample_from_nvvl(dataset, samples=10):
    indices = np.random.choice(len(dataset), size=samples, replace=False)
    return torch.cat([dataset[i]['input'].detach().to('cpu')[None] for i in indices])


def performance_plot(y_gt, y_pred):
    fig = pl.figure(0, figsize=(8, 4))
    x = np.arange(len(y_pred))
    pl.plot(x, y_pred)
    for start, end in connected_compoents(y_gt > 0.5):
        pl.axvspan(x[start], x[end], color='r', alpha=0.4)
    return fig


def create_unsupervised_trainer(model, optimizer, device=None, non_blocking=False):
    if device:
        model.to(device)

    def _update(engine, batch):
        optimizer.zero_grad()
        x = convert_tensor(batch['input'], device=device, non_blocking=non_blocking)
        loss, losses = model(x, retlosses=True)

        loss = loss.mean()
        loss.backward()
        optimizer.step()

        for key, val in losses.items():
            engine.state.metrics[key] = engine.state.metrics.get(key, 0) + val.mean().item()

        return loss.item()

    return ignite.engine.Engine(_update)


def get_loaders(traindir, batch_size, time_steps):
    training_videos = ['/home/users/daniel/data/shanghaitech/training/nvvl_videos/01_001.mp4']
    logger.info(f'Found {len(training_videos)} video files in training set.')
    testing_videos = ['/home/users/daniel/data/shanghaitech/testing/videos/01_0014.mp4']
    logger.info(f'Found {len(testing_videos)} video files in test set.')

    index_map = np.arange(16)
    processing = {
        'input': nvvl.ProcessDesc(
            #  scale_width=width, scale_height=height,
            normalized=True, dimension_order='cfhw',
            index_map=list(index_map))}
    frame_mask_dataset = FrameMaskDataset(
        '/home/users/daniel/data/shanghaitech/testing/test_frame_mask',
        index_map=index_map, video_paths=testing_videos)

    datasets = novelly.datasets.Split(
        nvvl.VideoDataset(
            training_videos, time_steps, processing=processing),
        nvvl.VideoDataset(
            testing_videos, time_steps,
            processing=processing, get_label=frame_mask_dataset.get_label))
    loaders = novelly.datasets.Split(
        nvvl.VideoLoader(datasets.train, batch_size=batch_size, shuffle=True, buffer_length=3),
        nvvl.VideoLoader(datasets.test, batch_size=batch_size, buffer_length=3))
    return loaders, frame_mask_dataset


def eval_test_dataset(predict_fn, loader, frame_mask_dataset, device):
    frames_dict = frame_mask_dataset.nr_frames
    groundtruth = {stem: torch.zeros(nr_frames)
                   for stem, nr_frames in frames_dict.items()}
    prediction = {stem: torch.zeros(nr_frames)
                  for stem, nr_frames in frames_dict.items()}
    counts = {stem: torch.zeros(nr_frames)
              for stem, nr_frames in frames_dict.items()}

    # FIXME Make this a ignite metric
    for data in loader:
        y_pred = predict_fn(data['input'].to(device)).to('cpu')

        for (mask_gt, idx, stem), mask_pd in zip(data['labels'], y_pred):
            groundtruth[stem][idx] += mask_gt
            prediction[stem][idx] += mask_pd
            counts[stem][idx] += 1

    for stem in groundtruth:
        sel = counts[stem].numpy() > 0
        groundtruth[stem] = (groundtruth[stem] / counts[stem]).numpy()[sel]
        prediction[stem] = (prediction[stem] / counts[stem]).numpy()[sel]
    return groundtruth, prediction


def ShanghaiTechExperiment(traindir, logdir, epochs):
    time_steps = 16
    height, width = (160, 256)
    batch_size = 4
    device = 'cuda:0'
    logdir = Path(logdir)
    log_interval = 10

    print('Setting up datasources')
    loaders, frame_mask_dataset = get_loaders(traindir, batch_size, time_steps)
    print('Setting up samples')
    sample_images = sample_from_nvvl(loaders.train.dataset, samples=3)

    print('Setting up models')
    encoder = novelly.ResidualVideoAE(
        input_shape=(time_steps, height, width),
        encoder_sizes=[8, 16, 32, 64, 64],
        decoder_sizes=[64, 32, 16, 8, 8],
        temporal_strides=[2, 2, 1, 1, 1],
        fc_sizes=[512, 64],
        color_channels=3,
        latent_activation=nn.Sigmoid())

    regressor = novelly.AutoregresionModule(
        dim=64,
        layer_sizes=[32, 32, 32, 32, 100],
        layer=novelly.AutoregressiveConvLayer)

    #  model = novelly.AutoregressiveVideoLoss(encoder, regressor, re_weight=1e-3, scale=1000)
    model = novelly.AutoregressiveVideoLoss(encoder, regressor, re_weight=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    summary_writer = novelly.datasets.Split(
        SummaryWriter(log_dir=str(logdir / 'train')),
        SummaryWriter(log_dir=str(logdir / 'val')))
    print('Lets go')
    trainer = create_unsupervised_trainer(model, optimizer, device=device, non_blocking=True)

    @trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        loader = engine.state.dataloader
        iteration = (engine.state.iteration - 1) % len(loader) + 1

        if iteration % log_interval == 0:
            print(f'Epoch[{engine.state.epoch}] '
                  f'Iteration[{iteration}/{len(loader)}] '
                  f'Loss: {engine.state.output:.2f}')
            summary_writer.train.add_scalar(
                'losses/total_run', engine.state.output, engine.state.iteration)

            for name, value in engine.state.metrics.items():
                summary_writer.train.add_scalar(
                    f'losses/{name}', value / log_interval, engine.state.iteration)
                engine.state.metrics[name] = 0

    @torch.no_grad()
    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
    def evaluate_model(engine):
        print('Running evaluation')
        #  if engine.state.epoch % 10 != 0: return
        model.eval()

        for i, original in enumerate(sample_images):
            reconstruction, = model.encoder(original[None].to(device)).to('cpu')
            original = make_grid(original.transpose(0, 1), nrow=original.size(1))
            recons = make_grid(reconstruction.transpose(0, 1), nrow=original.size(1))
            sample = make_grid([original, recons], nrow=1)
            summary_writer.train.add_image(f'train_images_{i}', sample,
                                           engine.state.epoch)

        roc_scores = []
        y_gts, y_preds = eval_test_dataset(
            model.predict, loaders.test, frame_mask_dataset, device)

        for key in y_gts:
            y_gt, y_pred = y_gts[key], y_preds[key]
            try:
                roc_scores.append(metrics.roc_auc_score(y_gt, y_pred))
            except ValueError:
                pass
            fig = performance_plot(y_gt, y_pred)
            fig = render_mpl_figure(fig)
            summary_writer.test.add_image(f'test_performance_{key}', fig,
                                          engine.state.epoch)

        roc_score = np.mean(roc_scores)
        summary_writer.test.add_scalar(f'metrics/avg_roc_auc', roc_score,
                                       engine.state.epoch)

    trainer.run(loaders.train, max_epochs=epochs)
