import json
import tempfile
from pathlib import Path

import click
import od
import torchvision as tv
from sklearn import metrics

from .classification_datasets import CIFAR10Experiment, MNISTExperiment
from .shanghai_tech import ShanghaiTechExperiment

WRITE_DIRECTORY = click.Path(file_okay=False, resolve_path=True, writable=True)


@click.group()
def experiments():
    pass


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


@experiments.command('shanghai-tech')
@click.option('--logdir', required=True, type=WRITE_DIRECTORY)
@click.option('--epochs', default=50, type=int)
def shanghai_tech(logdir, epochs):
    ShanghaiTechExperiment(
        traindir='/home/daniel/data/shanghaitech/training/h264',
        logdir=logdir, epochs=epochs).run()
