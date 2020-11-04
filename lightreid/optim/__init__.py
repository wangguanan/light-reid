"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

from .lr_scheduler import WarmupMultiStepLR, DelayedCosineAnnealingLR, WarmupCosineAnnealingLR
from easydict import EasyDict as edict
import torch

import lightreid


class Optimizer(object):

    KWARGS = ['fix_cnn_epochs']

    def __init__(self, optimizer, lr_scheduler, max_epochs, **kwargs):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_epochs = max_epochs
        for key, value in kwargs.items():
            assert key in Optimizer.KWARGS, 'expect {}, but got {}'.format(Optimizer.KWARGS, key)
            setattr(self, key, value)


__optimizer_factory = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
}

__lrscheduler_factory = {
    'warmup_multistep': WarmupMultiStepLR,
    'warmup_cosann': WarmupCosineAnnealingLR,
}


def build_optimizer(optimizer, lr_scheduler, **kwargs):
    """
    Args：
    """

    optimizer_cfg = edict(optimizer)
    lr_scheduler_cfg = edict(lr_scheduler)

    optimizer = __optimizer_factory[optimizer_cfg.pop('name')](**optimizer_cfg)
    lr_scheduler_cfg.optimizer = optimizer

    lr_scheduler = __lrscheduler_factory[lr_scheduler_cfg.pop('name')](**lr_scheduler_cfg)

    return Optimizer(optimizer=optimizer, lr_scheduler=lr_scheduler, **kwargs)



