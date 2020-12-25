from .generalize_mean_pooling import GeneralizedMeanPoolingP, GeneralizedMeanPooling
from .identity_pooling import IdentityPooling

import torch
import torch.nn as nn


__pooling_factory = {
    'avgpool': nn.AdaptiveAvgPool2d(1),
    'maxpool': nn.AdaptiveMaxPool2d(1),
    'gempool': GeneralizedMeanPooling(1),
    'gempoolp': GeneralizedMeanPoolingP(1),
    'identpool': IdentityPooling(),
}

class Clamp(nn.Module):

    def __init__(self, min, max):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, min=self.min, max=self.max)


def build_pooling(name, **kwargs):
    '''
    Example:
        pool = build_pooling('avgpool')
    '''
    assert name in __pooling_factory.keys(), \
        'expcet pooling name in {} but got {}'.format(__pooling_factory.keys(), name)

    layers = []
    pooling = __pooling_factory[name]
    layers.append(pooling)
    if 'clamp' in kwargs.keys():
        min, max = kwargs['clamp']
        clamp = Clamp(min=min, max=max)
        layers.append(clamp)

    return nn.Sequential(*layers)

