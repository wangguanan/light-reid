import torch
import torch.nn as nn

from .circle import Circle
from .arcface import ArcFace


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

