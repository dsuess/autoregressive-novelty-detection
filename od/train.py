import logging
from pathlib import Path

import click

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from .datasets import UnsupervisedImageDataset
from .encoders import ResidualAE
from time import time
from tensorboardX import SummaryWriter

READ_DIRECTORY = click.Path(exists=True, file_okay=False, resolve_path=True,
                            readable=True)
WRITE_DIRECTORY = click.Path(file_okay=False, resolve_path=True, writable=True)



@click.command('train')
@click.option('--images-dir', required=True, type=READ_DIRECTORY)
@click.option('--log-dir', required=True, type=WRITE_DIRECTORY)
def train_autoencoder(images_dir, log_dir):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    input_size = 32

    preprocessing = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop(input_size),
        transforms.ToTensor()
    ])

    dataset = UnsupervisedImageDataset(images_dir, transform=preprocessing)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8,
                        pin_memory=False)
    test_images = [dataset[i] for i in np.random.choice(len(dataset), size=100, replace=False)]
    model = ResidualAE((input_size, input_size), [64, 128, 256], [256, 64]).to(device)
    loss_fn = nn.MSELoss()
    optimizier = torch.optim.Adam(model.parameters())

    Path(log_dir).mkdir(exist_ok=True)
    summary_writer = SummaryWriter(log_dir)

    dummy_input = torch.Tensor(1, 3, 32, 32).to(device)
    summary_writer.add_graph(model, dummy_input)

    for epoch in tqdm(range(10)):

        loss_summary = torch.zeros(1, dtype=torch.float, device=device)
        for x in tqdm(loader):
            x = x.to(device)
            y = model(x)
            loss = loss_fn(x, y)
            loss_summary += loss

            optimizier.zero_grad()
            loss.backward()
            optimizier.step()

        summary_writer.add_scalar('loss', loss / len(dataset), epoch)

        model.eval()
        for n, img in enumerate(test_images):
            img_pred = model(img[None].to(device))[0]
            merged = make_grid([img.cpu(), img_pred.cpu()])
            summary_writer.add_image(f'image_{n}', merged, epoch)
        model.train()
