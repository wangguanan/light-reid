from .circle import Circle
from .generalize_mean_pooling import GeneralizedMeanPoolingP, GeneralizedMeanPooling
from .arcface import ArcFace


import torch.nn as nn

__pooling_factory = {
    'avgpool': nn.AdaptiveAvgPool2d(1),
    'maxpool': nn.AdaptiveMaxPool2d(1),
    'gempool': GeneralizedMeanPooling(1),
    'gempoolp': GeneralizedMeanPoolingP(1),
}

def build_pooling(name, **kwargs):
    '''
    Example:
        pool = build_pooling('avgpool')
    '''
    assert name in __pooling_factory.keys(), \
        'expcet pooling name in {} but got {}'.format(__pooling_factory.keys(), name)
    return __pooling_factory[name]


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