import functools as ft
import resource
import tempfile
from pathlib import Path

import click
import ignite
import torch
import torchvision as tv
import yaml
from ignite.engine import Events
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile

import novelly as nvly
from novelly.utils import Split, build_from_config, logger

WRITE_DIRECTORY = click.Path(file_okay=False, resolve_path=True, writable=True)


@click.group()
def main():
    pass


def build_data(cfg):
    ngpus = torch.cuda.device_count()
    datasets = Split(
        build_from_config(nvly.datasets, cfg['train'], train=True, download=True),
        build_from_config(nvly.datasets, cfg['valid'], train=False, download=True)
    )

    loaders = Split(
        datasets.train.LOADER(
            datasets.train, num_workers=ngpus * cfg['workers_per_gpu'],
            batch_size=ngpus * cfg['batch_size_per_gpu'], shuffle=True,
            drop_last=True),
        datasets.valid.LOADER(
            datasets.valid, num_workers=ngpus * cfg['workers_per_gpu'],
            batch_size=ngpus * cfg['batch_size_per_gpu'], shuffle=False)
    )

    return datasets, loaders


def make_example_images(autoencoder, sample_images, device=None):
    imgs_pred = autoencoder(sample_images.to(device))
    img_pairs = zip(sample_images, imgs_pred.cpu())
    img_merged = (tv.utils.make_grid(list(pair), nrow=1) for pair in img_pairs)
    all_images = tv.utils.make_grid(list(img_merged), nrow=len(sample_images))
    return all_images


def run_single_experiment(cfg, output_dir):
    ngpus = torch.cuda.device_count()
    device = 'cuda:0' if ngpus > 0 else 'cpu:0'
    autoencoder = build_from_config(nvly.encoders, cfg['model']['encoder'])
    model = build_from_config(
        nvly.autoregress, cfg['model']['regressor'], autoencoder=autoencoder)
    loss = ft.partial(nvly.autoregressive_loss, **cfg['model']['loss'])
    model.to(device)

    if ngpus > 1:
        cfg['train']['optimizer']['lr'] *= ngpus
        model = torch.nn.DataParallel(model)

    datasets, loaders = build_data(cfg['data'])
    optimizer = build_from_config(
        torch.optim, cfg['train']['optimizer'], params=model.parameters())
    scheduler = build_from_config(
        (nvly.lr_scheduler, torch.optim.lr_scheduler), cfg['train']['scheduler'],
        optimizer=optimizer)

    # Create evaluator
    metrics = {'roc_auc': nvly.engine.RocAucScore()}
    evaluator = ignite.engine.create_supervised_evaluator(
        model, metrics=metrics, device=device, non_blocking=True)
    summary_writer = SummaryWriter(output_dir)

    # Create trainer
    trainer = nvly.engine.create_unsupervised_trainer(
        model, optimizer, loss, device=device, non_blocking=True)
    saver_args = {
        'model': model, 'ckpt_dir': output_dir, 'optimizer': optimizer,
        'scheduler': scheduler}
    trainer.add_event_handler(
        Events.STARTED, nvly.engine.restore_latest_checkpoint(**saver_args))

    update_scheduler = nvly.engine.step_lr_scheduler(
        optimizer, scheduler, on_epoch=False, summary_writer=summary_writer,
        verbose=False)
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, nvly.engine.every_n(10, update_scheduler))
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        nvly.engine.log_iterations_per_second(n=50, summary_writer=summary_writer))

    ckpt_every_n_epochs = cfg['train'].get('ckpt_every_n_epochs', None)
    callback = nvly.engine.every_n(
        ckpt_every_n_epochs, callback=nvly.engine.save_checkpoint(**saver_args),
        on_epoch=True)
    if ckpt_every_n_epochs is not None:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, callback)

    @trainer.on(Events.ITERATION_COMPLETED)
    @nvly.engine.every_n(10)
    def log_training_loss(engine):
        prefix = nvly.engine.get_log_prefix(engine)
        msgs = ', '.join(
            [f'{name}={value:.3f}' for name, value in engine.state.metrics.items()])
        print(f'{prefix} Losses: {msgs}')

        for name, val in engine.state.metrics.items():
            summary_writer.add_scalar(
                f'losses/{name}', val, engine.state.iteration)
            engine.state.metrics[name] = 0
        engine.state.avg_counter = 0

    @trainer.on(Events.EPOCH_STARTED)
    def reset_metrics(engine):
        engine.state.avg_counter = 0
        for name in engine.state.metrics:
            engine.state.metrics[name] = 0

    example_images = {
        'known': datasets.valid.sample_images(10, should_be_known=True),
        'unknown': datasets.valid.sample_images(10, should_be_known=False)
    }
    best_score = 0

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_evaluation_results(engine):
        model.eval()
        for name, imgs in example_images.items():
            imgs = make_example_images(autoencoder, imgs, device=device)
            summary_writer.add_image(name, imgs, engine.state.iteration)

        evaluator.run(loaders.valid)
        metrics = evaluator.state.metrics

        prefix = nvly.engine.get_log_prefix(engine)
        msgs = ', '.join(
            [f'{name}: {value:.3f}' for name, value in metrics.items()])
        print(f'{prefix} {msgs}')

        for name, value in metrics.items():
            summary_writer.add_scalar(
                f'metrics/{name}', value, engine.state.iteration)

        nonlocal best_score
        best_score = max(best_score, metrics['roc_auc'])

    trainer.run(loaders.train, max_epochs=cfg['train']['epochs'])
    return best_score


@main.command(name='single-experiment')
@click.option('--config-file', '-c', required=True, type=click.Path(dir_okay=False, exists=True))
@click.option('--output-dir', '-o', required=True, type=click.Path(file_okay=False, writable=True))
def single_experiment(config_file, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    config_file = Path(config_file)
    with open(config_file) as buf:
        cfg = yaml.load(buf, Loader=yaml.FullLoader)
    if not (output_dir / config_file.name).exists():
        copyfile(config_file, output_dir / config_file.name)
    best_score = run_single_experiment(cfg, output_dir)
    print(best_score)


@main.command(name='multiple-experiments')
@click.option('--config-file', '-c', required=True, type=click.Path(dir_okay=False, exists=True))
@click.option('--output-dir', '-o', required=True, type=click.Path(file_okay=False, writable=True))
def multiple_experiments(config_file, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    config_file = Path(config_file)
    with open(config_file) as buf:
        cfg = yaml.load(buf, Loader=yaml.FullLoader)
    if not (output_dir / config_file.name).exists():
        copyfile(config_file, output_dir / config_file.name)

    result = []
    for experiment_configurations in cfg['experiment_configurations']:
        local_cfg = cfg.copy()
        local_cfg.pop('experiment_configurations')
        positive_classes  = experiment_configurations['positive_classes']
        local_cfg['data']['train']['positive_classes'] = positive_classes
        local_cfg['data']['valid']['positive_classes'] = positive_classes
        local_output_dir = output_dir / f'with_{"_".join(str(s) for s in positive_classes)}'
        local_output_dir.mkdir(exist_ok=True)
        best_score = run_single_experiment(local_cfg, local_output_dir)
        result.append(best_score)

    print(result)


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
