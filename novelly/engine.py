import functools as ft
import os
from pathlib import Path
from time import time

import torch
import ignite
from ignite.engine.engine import Engine
from ignite.utils import convert_tensor

from novelly.utils import logger
from sklearn.metrics import roc_auc_score

__all__ = ['create_unsupervised_trainer']


def _prepare_batch(x, device=None, non_blocking=False):
    return convert_tensor(x, device=device, non_blocking=non_blocking)


def create_unsupervised_trainer(model, optimizer, loss_fn, device=None, non_blocking=False,
                                prepare_batch=_prepare_batch):
    if device:
        model.to(device)

    def _update(engine, x):
        model.train()
        optimizer.zero_grad()
        x = prepare_batch(x, device=device, non_blocking=non_blocking)
        y = model(x, return_reconstruction=True)
        loss, loss_items = loss_fn(y, x, retlosses=True)
        loss.backward()
        optimizer.step()

        n = getattr(engine.state, 'avg_counter', 0)
        for key, val in loss_items.items():
            new_val = n * engine.state.metrics.get(key, 0) + val.item()
            engine.state.metrics[key] = new_val / (n + 1)

        engine.state.avg_counter = n + 1
        return loss

    return Engine(_update)


def get_log_prefix(engine):
    n_iterations = len(engine.state.dataloader)
    return (f'[Epoch {engine.state.epoch}/{engine.state.max_epochs}] '
            f'[Iteration {engine.state.iteration % n_iterations}/{n_iterations}]')


def restore_latest_checkpoint(model, ckpt_dir, glob='*.pt', verbose=True,
                              **kwargs):
    ckpt_dir = Path(ckpt_dir)
    try:
        module = model.module
    except AttributeError:
        module = model

    def func(engine):
        try:
            latest_ckpt_path = max(ckpt_dir.glob(glob), key=os.path.getctime)
        except ValueError:
            return

        checkpoint = torch.load(latest_ckpt_path)
        engine.state.epoch = checkpoint['epoch']
        engine.state.iteration = checkpoint['iteration']
        module.load_state_dict(checkpoint['model'])

        for name, item in kwargs.items():
            item.load_state_dict(checkpoint[name])
        if verbose:
            print(f'{get_log_prefix(engine)} Restored from {latest_ckpt_path}')

    return func


def save_checkpoint(model, ckpt_dir, template='checkpoint_{epoch}.pt',
                    verbose=True, **kwargs):
    ckpt_dir = Path(ckpt_dir)
    try:
        module = model.module
    except AttributeError:
        module = model

    def func(engine):
        state = {
            'epoch': engine.state.epoch,
            'iteration': engine.state.iteration,
            'model': module.state_dict()}

        for name, item in kwargs.items():
            state[name] = item.state_dict()

        ckpt_path = ckpt_dir / template.format(epoch=engine.state.epoch,
                                               iteration=engine.state.iteration)
        torch.save(state, ckpt_path)
        if verbose:
            print(f'{get_log_prefix(engine)} Saved checkpoint to {ckpt_path}')

    return func


def step_lr_scheduler(optimizer, scheduler, on_epoch=True, summary_writer=None,
                      verbose=True):

    def func(engine):
        scheduler.step(engine.state.epoch if on_epoch else engine.state.iteration)
        learning_rates = [g['lr'] for g in optimizer.param_groups]
        if verbose:
            print(f'{get_log_prefix(engine)} Set learning rates to {learning_rates}')

        if summary_writer is not None:
            for n, param_group in enumerate(optimizer.param_groups):
                summary_writer.add_scalar(
                    f'stats/lr_{n}', param_group['lr'], engine.state.iteration)

    return func


def log_iterations_per_second(n=10, summary_writer=None, verbose=True):
    def func(engine):
        loader = engine.state.dataloader
        counter = (engine.state.iteration - 1) % len(loader) + 1
        if counter % n != 0:
            return

        last_called = getattr(engine.state, 'last_called', None)
        if (last_called is not None) and counter >= n:
            runtime = time() - last_called
            it_per_s = n / runtime

            if verbose:
                print(f'{get_log_prefix(engine)} {it_per_s:.2f} it/s')
            if summary_writer is not None:
                summary_writer.add_scalar(
                    f'stats/it_per_s', it_per_s, engine.state.iteration)

        engine.state.last_called = time()
    return func


def every_n(n, callback=None, on_epoch=False):
    def func(engine, callback=None):
        if on_epoch:
            counter = engine.state.epoch
        else:
            loader = engine.state.dataloader
            counter = (engine.state.iteration - 1) % len(loader) + 1

        if counter % n == 0:
            callback(engine)

    return ft.partial(func, callback=callback) if callback is not None \
        else (lambda c: ft.partial(func, callback=c))


class RocAucScore(ignite.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.y_t = []
        self.y_p = []

    def reset(self):
        self.y_t = []
        self.y_p = []

    def update(self, output):
        # "-" because autoreg. losses approximates neg. log. likelihood
        self.y_p += list(-output[0].detach().to('cpu').numpy())
        self.y_t += list(output[1].detach().to('cpu').numpy())

    def compute(self):
        return roc_auc_score(self.y_t, self.y_p)
