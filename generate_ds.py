from typing import Literal, Tuple, Optional, Callable
import threading
import random

from PIL import Image

import numpy as np
import re
from pathlib import Path
from skimage import morphology

import rasterio
import rasterio.transform, rasterio.windows
import math
from concurrent import futures


class AP_jp2:
    PRED_PATTERN = re.compile(r'(?P<wnid>n\d+) (?P<x0>\d+) (?P<y0>\d+) (?P<x1>\d+) (?P<y1>\d+)')

    def __init__(
        self,
        root: str | Path,
        tile_size: int,
        thickness=5,
        x_id=3,
        y_id=1,
        downsample=1,
        step: Optional[int] = None,
        split: Literal['train'] | Literal['val'] = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        assert split in ['train', 'val']
        self.root = Path(root)
        assert self.root.is_dir()
        self.split = split
        self.train = split == 'train'
        self.transform = transform
        self.target_transform = target_transform
        self.tile_size = tile_size
        assert math.log2(downsample).is_integer(), 'Expected power of 2 downsample'
        self.downsample = downsample
        if step is None:
            step = tile_size
        self.step = step
        # self.target_transform = target_transform

        self.x_slide: rasterio.DatasetReader = rasterio.open(self.root / f'sample_{thickness}um_AP_{x_id}.jp2')
        self.y_slide: rasterio.DatasetReader = rasterio.open(self.root / f'sample_{thickness}um_AP_{y_id}.jp2')

        self.x_slide_lock = threading.Lock()
        self.y_slide_lock = threading.Lock()

        height = min(self.x_slide.height, self.y_slide.height)
        width = min(self.x_slide.width, self.y_slide.width)

        y_positions = [i * self.step for i in range(height // self.downsample // self.step)]
        y_positions = [y for y in y_positions if (y > height // self.downsample // 2) == self.train]
        x_positions = [i * self.step for i in range(width // self.downsample // self.step)]
        self.tile_positions = [(x, y) for y in y_positions for x in x_positions]

    def __len__(self):
        return len(self.tile_positions)

    def get_tile_pair(
        self,
        x: int,
        y: int,
        tile_size: Optional[int] = None,
        downsample: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        tile_size = tile_size if tile_size is not None else self.tile_size
        downsample = downsample if downsample is not None else self.downsample

        window = [x, y + tile_size, x + tile_size, y]
        window = [w * downsample for w in window]
        try:
            with self.x_slide_lock:
                x_tile = self.x_slide.read(
                    (1, 2, 3),
                    window=rasterio.windows.from_bounds(*window, transform=self.x_slide.transform),
                    out_shape=(tile_size, tile_size),
                )
            with self.y_slide_lock:
                y_tile = self.y_slide.read(
                    (1, 2, 3),
                    window=rasterio.windows.from_bounds(*window, transform=self.y_slide.transform),
                    out_shape=(tile_size, tile_size),
                )
        except rasterio.RasterioIOError:
            # x_tile, y_tile = self.get_tile_pair(*self.tile_positions[random.randrange(0, len(self))])
            x_tile = np.full((tile_size, tile_size, 3), 158)
            y_tile = np.full((tile_size, tile_size, 3), 158)
            # x_tile = np.zeros((3, tile_size, tile_size), dtype=np.uint8)
            # y_tile = np.zeros((3, tile_size, tile_size), dtype=np.uint8)
        return x_tile, y_tile

    def __getitem__(self, idx):
        # print(f'{self.tile_positions[idx]=}')
        x, y = self.get_tile_pair(*self.tile_positions[idx])
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y


class RemoveBlack:
    def __init__(self, initial_fill: Tuple[int, int, int] = (158, 158, 158)) -> None:
        self.fill_value = np.array(list(initial_fill), dtype=np.float32)
        self.blank_count = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        blacks = np.all(x == 0, axis=0)
        if np.all(blacks):
            return x
            return np.full_like(x, self.fill_value.astype(np.uint8)[:, np.newaxis, np.newaxis])
        elif np.any(blacks):
            blacks = morphology.binary_dilation(blacks, np.ones((5, 5)))
            border = morphology.binary_dilation(blacks, np.ones((5, 5)))
            border = border & ~blacks
            blacks, border, x = np.broadcast_arrays(blacks, border, x)
            # return blacks
            med = np.median(x[border], axis=0)
            self.fill_value = self.blank_count * self.fill_value + med
            self.blank_count += 1
            self.fill_value /= self.blank_count
            x = np.where(blacks, med.astype(np.uint8), x)
            return x
        else:
            return x


ds = AP_jp2(
    '/users/40390351/sharedscratch/AP_jp2s',
    286,
    step=256,
    downsample=1,
    split='train',
    transform=RemoveBlack(),
    target_transform=RemoveBlack(),
)


def save_pair(i: int):
    pair = ds[i]
    for name, x in zip(['A', 'B'], pair):
        blacks = np.all(x == 0, axis=0)
        if np.all(blacks):
            print('Black. Skipping...')
            continue
        else:
            if np.any(blacks):
                blacks = morphology.binary_dilation(blacks, np.ones((5, 5)))
                border = morphology.binary_dilation(blacks, np.ones((5, 5)))
                border = border & ~blacks
                blacks, border, x = np.broadcast_arrays(blacks, border, x)
                # return blacks
                med = np.median(x[border], axis=0)
                x = np.where(blacks, med.astype(np.uint8), x)
            path = Path(f'datasets/mouse_ds1/train{name}')
            path.mkdir(exist_ok=True, parents=True)
            Image.fromarray(x.transpose((1, 2, 0))).save(path / f'{i}.png')


with futures.ThreadPoolExecutor() as pool:
    futs = []
    for i in range(len(ds)):
        futs.append(pool.submit(save_pair, i))

    for i, fut in enumerate(futures.as_completed(futs)):
        fut.result()
        print(i + 1, 'completed')
