from .circle import Circle
from .generalize_mean_pooling import GeneralizedMeanPoolingP, GeneralizedMeanPooling
from .arcface import ArcFace

import torch
import torch.nn as nn

__pooling_factory = {
    'avgpool': nn.AdaptiveAvgPool2d(1),
    'maxpool': nn.AdaptiveMaxPool2d(1),
    'gempool': GeneralizedMeanPooling(1),
    'gempoolp': GeneralizedMeanPoolingP(1),
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



__classifier_factory = {
    'linear': nn.Linear,
    'circle': Circle,
    'arcface': ArcFace,
}

def build_classifier(name, in_dim, out_dim, **kwargs):
    '''
    Example:
        classifier = build_classifier('linear', 2048, 100)
    '''
    assert name in __classifier_factory.keys(), \
        'expect classifier in {} but got {}'.format(__classifier_factory.keys(), name)
    if name in ['linear']:
        return __classifier_factory[name](in_dim, out_dim)
    elif name in ['circle', 'arcface']:
        assert 'scale' in kwargs.keys() and 'margin' in kwargs.keys(), \
            '{} require parameters scale and margin, but not given'.format(name)
        scale = kwargs['scale']
        margin = kwargs['margin']
        return __classifier_factory[name](in_dim, out_dim, scale=scale, margin=margin)
    else:
        raise AssertionError