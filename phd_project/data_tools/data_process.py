# Copyright (c) OpenMMLab. All rights reserved.
import platform

from mmcv.utils import build_from_cfg

from mmdet.datasets import DATASETS as MMDET_DATASETS
from mmdet.datasets.builder import _concat_dataset

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def build_dataset(cfg, default_args=None):
    dataset = build_from_cfg(cfg, MMDET_DATASETS, default_args)
    return dataset
