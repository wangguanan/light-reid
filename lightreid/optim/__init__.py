"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

from .lr_scheduler import WarmupMultiStepLR, DelayedCosineAnnealingLR, WarmupCosineAnnealingLR


class Optimizer(object):

    KWARGS = ['fix_cnn_epochs']

    def __init__(self, optimizer, lr_scheduler, max_epochs, **kwargs):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_epochs = max_epochs
        for key, value in kwargs.items():
            assert key in Optimizer.KWARGS, 'expect {}, but got {}'.format(Optimizer.KWARGS, key)
            setattr(self, key, value)



# optimizer = torch.optim.Adam(model.parameters(), lr=0.00035, weight_decay=5e-4)
# lr_scheduler = lightreid.optim.WarmupMultiStepLR(optimizer, milestones=args.milestones, gamma=0.1, warmup_factor=0.01, warmup_epochs=args.warmup_epochs)
# optimizer = lightreid.optim.Optimizer(optimizer=optimizer, lr_scheduler=lr_scheduler, max_epochs=args.total_epochs)
#
# import torch
# import lightreid
#
# __optimizer_factory = {
#     'adam': torch.optim.Adam,
#     'sgd': torch.optim.SGD,
# }
#
# __lrscheduler_factory = {
#     'warmup_multistep': lightreid.optim.WarmupMultiStepLR,
#     'warmup_cosann': lightreid.optim.WarmupCosineAnnealingLR,
# }
#
# build_optimizer(
#     optimizer={'name': 'adam', 'parameters': model.parameters(), 'lr': 0.00035, 'weight': 5e-4},
#     lr_scheduler={'name': 'warmupo_multistep', 'milestones': [40, 70], 'warmup_factor': 0.01, 'earmup_epochs': 10}
# )
#
# def build_optimizer(optimizer, lr_scheduler, **kwargs):
#


