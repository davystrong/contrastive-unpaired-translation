from typing import Tuple, Optional, Callable
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
from models import networks
import torch
from skimage import transform
from models.cut_model import CUTModel
from argparse import ArgumentParser, Namespace
import sys
import torchvision.transforms as transforms

from options.base_options import BaseOptions
from models import create_model

class AP_jp2:
    PRED_PATTERN = re.compile(r'(?P<wnid>n\d+) (?P<x0>\d+) (?P<y0>\d+) (?P<x1>\d+) (?P<y1>\d+)')

    def __init__(self, root: str | Path, tile_size: int, thickness=5, x_id=3, y_id=1, downsample=1, step: Optional[int]=None, split='train', transform: Optional[Callable]=None, target_transform: Optional[Callable]=None) -> None:
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

        self.x_slide: rasterio.DatasetReader = rasterio.open(self.root/f'sample_{thickness}um_AP_{x_id}.jp2')
        self.y_slide: rasterio.DatasetReader = rasterio.open(self.root/f'sample_{thickness}um_AP_{y_id}.jp2')

        self.x_slide_lock = threading.Lock()
        self.y_slide_lock = threading.Lock()

        height = min(self.x_slide.height, self.y_slide.height)
        width = min(self.x_slide.width, self.y_slide.width)

        y_positions = [i*self.step for i in range(height // self.downsample // self.step)]
        y_positions = [y for y in y_positions if (y > height // self.downsample // 2) == self.train]
        x_positions = [i*self.step for i in range(width // self.downsample // self.step)]
        self.tile_positions = [(x, y) for y in y_positions for x in x_positions]

    def __len__(self):
        # return 2**12
        return len(self.tile_positions)

    def get_tile_pair(self, x: int, y: int, tile_size: Optional[int]=None, downsample: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]:
        tile_size = tile_size if tile_size is not None else self.tile_size
        downsample = downsample if downsample is not None else self.downsample

        window = [x, y+tile_size, x+tile_size, y]
        window = [w*downsample for w in window]
        try:
            with self.x_slide_lock:
                x_tile = self.x_slide.read((1, 2, 3), window=rasterio.windows.from_bounds(*window, transform=self.x_slide.transform), out_shape=(tile_size, tile_size))
            with self.y_slide_lock:
                y_tile = self.y_slide.read((1, 2, 3), window=rasterio.windows.from_bounds(*window, transform=self.y_slide.transform), out_shape=(tile_size, tile_size))
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
    def __init__(self, initial_fill: Tuple[int, int, int]=(158, 158, 158)) -> None:
        self.fill_value = np.array(list(initial_fill), dtype=np.float32)
        self.blank_count = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        blacks = np.all(x == 0, axis=0)
        if np.all(blacks):
            return np.full_like(x, self.fill_value.astype(np.uint8)[:, np.newaxis, np.newaxis])
        elif np.any(blacks):
            blacks = morphology.binary_dilation(blacks, np.ones((5, 5)))
            border = morphology.binary_dilation(blacks, np.ones((5, 5)))
            border = border & ~blacks
            blacks, border, x = np.broadcast_arrays(blacks, border, x)
            # return blacks
            med = np.median(x[border], axis=0)
            self.fill_value = self.blank_count*self.fill_value + med
            self.blank_count += 1
            self.fill_value /= self.blank_count
            x = np.where(blacks, med.astype(np.uint8), x)
            return x
        else:
            return x
        

trans_black = RemoveBlack()
target_trans_black = RemoveBlack()
# ds = AP_jp2('/users/40390351/sharedscratch/AP_jp2s', 286, step=256, downsample=4, split='train', transform=trans_black, target_transform=target_trans_black)
ds = AP_jp2('/users/40390351/sharedscratch/AP_jp2s', 288, step=256, downsample=4, split='train', transform=trans_black, target_transform=target_trans_black)
device = torch.device('cuda')

# model = networks.define_G(3, 3, 64, 'resnet_9blocks', 'instance', True, 'xavier', 0.02, False, False, [0], None)
# state_dict = torch.load('checkpoints/mouse_CUT_v1/215_net_G.pth', map_location=str(device))
# for key in list(state_dict.keys()):
#     if '.5.' in key:
#         new_key = '.6.'.join(key.split('.5.'))
#         state_dict[new_key] = state_dict[key]
#         del state_dict[key]
# model.load_state_dict(state_dict)

opts = Namespace(
    gpu_ids=[0],
    isTrain=False,
    checkpoints_dir='./checkpoints',
    dataroot='./datasets/mouse',
    name='mouse_CUT_v1',
    model='cut',
    CUT_mode='CUT',
    normG='instance',
    preprocess='resize_and_crop',
    nce_layers='0,4,8,12,16',
    nce_idt=False,
    input_nc=3,
    output_nc=3,
    ngf=64,
    ndf=64,
    netG='resnet_9blocks',
    netD='basic',
    no_dropout=False,
    init_type='xavier',
    init_gain=0.02,
    no_antialias=False,
    no_antialias_up=False,
    netF='mlp_sample',
    netF_nc=256,
    n_layers_D=3,
    normD='instance',
    gan_mode='lsgan'
)

cut_model = create_model(opts)
cut_model.load_networks(215)
model = cut_model.netG
# model.eval()
model_lock = threading.Lock()

def run_model(i: int, batch_size: int) -> np.ndarray:
    batch = np.stack([ds[j][0] for j in range(i, i+batch_size)])
    batch = torch.from_numpy(batch).to(device, torch.float32)
    batch = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(batch)
    with model_lock:
        with torch.no_grad():
            pred_batch = model(batch).cpu().detach().numpy()
    pred_batch = np.transpose(pred_batch, (0, 2, 3, 1))
    pred_batch = pred_batch[:, 16:-16, 16:-16]
    resized_batch = transform.downscale_local_mean(pred_batch, (1, 16, 16, 1))
    resized_batch = ((resized_batch + 1)*255.0/2).astype(np.uint8)
    return resized_batch

BATCH_SIZE = 4

with futures.ThreadPoolExecutor() as pool:
    outputs = map(lambda i: run_model(i, BATCH_SIZE), range(0, len(ds), BATCH_SIZE))
    outputs = (x for batch in outputs for x in batch)
    out_img = None
    in_img = None
    gt_img = None
    try:
        with Image.open('ds_output.png') as img:
            out_img = np.array(img)
        with Image.open('ds_input.png') as img:
            in_img = np.array(img)
        with Image.open('ds_gt.png') as img:
            gt_img = np.array(img)
    except Exception:
        pass
    downsampled_positions = [(x//16, y//16) for x, y in ds.tile_positions]
    for i, ((x, y), tile) in enumerate(zip(downsampled_positions, outputs)):
        if out_img is None:
            out_img = np.zeros((tile.shape[0] + max(y for x, y in downsampled_positions), tile.shape[1] + max(x for x, y in downsampled_positions), 3), dtype=np.uint8)
        if in_img is None:
            in_img = np.zeros((tile.shape[0] + max(y for x, y in downsampled_positions), tile.shape[1] + max(x for x, y in downsampled_positions), 3), dtype=np.uint8)
        if gt_img is None:
            gt_img = np.zeros((tile.shape[0] + max(y for x, y in downsampled_positions), tile.shape[1] + max(x for x, y in downsampled_positions), 3), dtype=np.uint8)
        # padding = np.maximum(np.array([y, x, 0]) + tile.shape - img.shape, 0)
        # padding = np.stack([np.zeros_like(padding), padding], axis=-1)
        # if np.any(padding > 0):
        #     img = np.pad(img, padding)
        out_img[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
        in_img[y:y+tile.shape[0], x:x+tile.shape[1]] = transform.downscale_local_mean(np.transpose(ds[i][0], (1, 2, 0))[16:-16, 16:-16], (16, 16, 1))
        gt_img[y:y+tile.shape[0], x:x+tile.shape[1]] = transform.downscale_local_mean(np.transpose(ds[i][1], (1, 2, 0))[16:-16, 16:-16], (16, 16, 1))
        print(i+1, 'completed')
        # print(tile.shape)

        # Image.fromarray(tile).save('sample.jpg')
        Image.fromarray(out_img).save(f'ds_output.png')
        Image.fromarray(in_img).save(f'ds_input.png')
        Image.fromarray(gt_img).save(f'ds_gt.png')