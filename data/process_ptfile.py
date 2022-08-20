import datetime
import os
import traceback
import zipfile
from argparse import Namespace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

import sys
sys.path.insert(0, '/home/maltese/PycharmProjects/mega-nerf')

import sys
sys.path.insert(0, '/home/hyunjin/PycharmProjects/mega-nerf')

from mega_nerf.misc_utils import main_tqdm, main_print
from mega_nerf.opts import get_opts_base
from mega_nerf.ray_utils import get_ray_directions, get_rays


def _get_mask_opts() -> Namespace:
    parser = get_opts_base()

    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--grid_dim', nargs='+', type=int, required=True)
    parser.add_argument('--ray_samples', type=int, default=1000)
    parser.add_argument('--ray_chunk_size', type=int, default=48 * 1024)
    parser.add_argument('--dist_chunk_size', type=int, default=64 * 1024 * 1024)
    parser.add_argument('--resume', default=False, action='store_true')

    return parser.parse_known_args()[0]


@torch.inference_mode()
def main(hparams: Namespace) -> None:
    assert hparams.ray_altitude_range is not None
    output_path = Path(hparams.output)

    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(0, hours=24))
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        rank = int(os.environ['RANK'])
        if rank == 0:
            output_path.mkdir(parents=True, exist_ok=hparams.resume)
        dist.barrier()
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        output_path.mkdir(parents=True, exist_ok=hparams.resume)
        rank = 0
        world_size = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = Path(hparams.dataset_path)
    coordinate_info = torch.load(dataset_path / 'coordinates.pt') # 여기 정보 확인해야해.
    origin_drb = coordinate_info['origin_drb']
    pose_scale_factor = coordinate_info['pose_scale_factor']

    ray_altitude_range = [(x - origin_drb[0]) / pose_scale_factor for x in hparams.ray_altitude_range]

    metadata_paths = list((dataset_path / 'train' / 'metadata').iterdir()) \
                     + list((dataset_path / 'val' / 'metadata').iterdir())

    camera_positions = torch.cat([torch.load(x, map_location='cpu')['c2w'][:3, 3].unsqueeze(0) for x in metadata_paths])
    main_print('Number of images in dir: {}'.format(camera_positions.shape))



if __name__ == '__main__':
    main(_get_mask_opts())
