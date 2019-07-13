import json
import resource


import tempfile
from pathlib import Path

import click
import torchvision as tv

import novelly as nvly
from sklearn import metrics

WRITE_DIRECTORY = click.Path(file_okay=False, resolve_path=True, writable=True)


@click.group()
def main():
    pass


@main.command(name='mnist')
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
        logdir.mkdir(exist_ok=True, parent=True)
        result = dict()

        for i in range(10):
            datasets = nvly.datasets.MNIST.load_split(
                download_dir, {i}, download=True, transforms=transforms)
            experiment = nvly.MNISTExperiment(
                datasets=datasets, logdir=str(logdir / f'only_{i}'), epochs=epochs)
            experiment.run(keep_every_ckpt=False)

            losses = experiment.compute_eval_losses()
            roc_score = metrics.roc_auc_score(losses.test_known, -losses.test)
            result[i] = roc_score

        with open(logdir / 'result.json', 'w') as buf:
            json.dump(result, buf)

    else:
        datasets = nvly.datasets.MNIST.load_split(
            download_dir, {1, 2, 3}, download=True, transforms=transforms)
        nvly.MNISTExperiment(datasets=datasets, logdir=logdir, epochs=epochs).run()


@main.command(name='cifar10')
@click.option('--logdir', required=True, type=WRITE_DIRECTORY)
@click.option('--download-dir', required=False, type=WRITE_DIRECTORY)
@click.option('--epochs', default=50, type=int)
@click.option('--batch', is_flag=True)
def cifar10(logdir, download_dir, epochs, batch):
    if download_dir is None:
        download_dir = Path(tempfile.gettempdir()) / 'cifar10'

    if batch:
        logdir = Path(logdir)
        logdir.mkdir(exist_ok=True, parent=True)
        result = dict()

        for i in range(10):
            datasets = nvly.datasets.CIFAR10.load_split(
                download_dir, {i}, download=True)
            experiment = nvly.CIFAR10Experiment(
                datasets=datasets, logdir=logdir / f'only_{i}', epochs=epochs)
            experiment.run(keep_every_ckpt=False)

            losses = experiment.compute_eval_losses()
            roc_score = metrics.roc_auc_score(losses.test_known, -losses.test)
            result[i] = roc_score

        with open(logdir / 'result.json', 'w') as buf:
            json.dump(result, buf)

    else:
        datasets = nvly.datasets.CIFAR10.load_split(
            download_dir, {1, 2, 3}, download=True)
        nvly.CIFAR10Experiment(datasets=datasets, logdir=logdir, epochs=epochs).run()


@main.command('shanghai-tech')
@click.option('--logdir', required=True, type=WRITE_DIRECTORY)
@click.option('--epochs', default=50, type=int)
def shanghai_tech(logdir, epochs):
    nvly.ShanghaiTechExperiment(
        traindir='/home/daniel/data/shanghaitech/training/h264',
        logdir=logdir, epochs=epochs)


if __name__ == '__main__':
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    main()
