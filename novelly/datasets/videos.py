import functools as ft
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..utils import logger


class FrameMaskDataset:

    def __init__(self, datadir, index_map, video_paths=None):
        self.datadir = Path(datadir)

        index_map = np.array(index_map, dtype=np.int64)
        index_map = np.argsort(index_map)[sum(index_map < 0):]
        # make sure index_map is contiguous with each index only once
        assert len(index_map) == max(index_map) + 1
        self.index_map = torch.from_numpy(index_map)

        if video_paths is None:
            files = self.datadir.glob('*.npy')
        else:
            files = [self.datadir / f'{Path(s).stem}.npy' for s in video_paths]

        # 1 - x since y_gt = 1.0 should equal known frame
        self.frame_masks = {s.stem: torch.Tensor(1 - np.load(s)) for s in files}
        logger.info(f'Found {len(self.frame_masks)} frame masks in {self.datadir}')

    def get_label(self, filename, frame_id, _):
        stem = Path(filename).stem
        indices = frame_id + self.index_map
        # "1-" since class 1 = novel
        mask = 1 - self.frame_masks[stem][indices]
        return mask, indices, stem

    @property
    def nr_frames(self):
        return {key: len(val) for key, val in self.frame_masks.items()}
