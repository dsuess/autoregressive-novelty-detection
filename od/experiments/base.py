import abc
import os
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as pl
import torch
from od.utils import logger
from tensorboardX import SummaryWriter
from tqdm import tqdm


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
