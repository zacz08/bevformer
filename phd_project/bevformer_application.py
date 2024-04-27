# ---------------------------------------------
#  Modified by Cheng Zhang
# ---------------------------------------------

import torch
import argparse
import os

from mmcv import Config
from mmdet3d.models import build_model
from phd_project.data_tools import data_process
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmcv.runner import load_checkpoint, init_dist
from mmcv.parallel import MMDistributedDataParallel


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def get_BEVFormer_output(model, data):
    """
    Get the return of BEVFormer for each input batch
    """
    model.eval()
    with torch.no_grad():
        """
        model return bbox_results, which is a list of predicted bounding box
        """
        result = model(return_loss=False, rescale=True, **data)

    return result


def main():
    
    ########### test zone ###########
    
    ########### test zone end ###########
    
    args = parse_args()

    # get BEVFormer configuration
    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    distributed = True
    init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # dataset = build_dataset(cfg.data.train)
    dataset = data_process.build_dataset(cfg.data.test)


    ########### test zone ###########

    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    nusc_can = NuScenesCanBus(dataroot='/home/zc/datasets/nuscenes')
    
    ########### test zone end ###########


    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfpose_listg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    
    
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # load model with DDP mode
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False
    )

    # get the first batch from dataloader by iterator
    data = next(iter(data_loader))

    res = get_BEVFormer_output(model, data)
    print("result:", res)


if __name__ == '__main__':
    main()