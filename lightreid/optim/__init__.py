"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

from .lr_scheduler import WarmupMultiStepLR, DelayedCosineAnnealingLR

class Optimizer(object):

    KWARGS = ['fix_cnn_epochs']

    def __init__(self, optimizer, lr_scheduler, max_epochs, **kwargs):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_epochs = max_epochs
        for key, value in kwargs.items():
            assert key in Optimizer.KWARGS, 'expect {}, but got {}'.format(Optimizer.KWARGS, key)
            setattr(self, key, value)

