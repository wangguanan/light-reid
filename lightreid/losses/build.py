"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import torch
from easydict import EasyDict as edict

from lightreid.utils import Registry
LOSSes_REGISTRY = Registry('losses')

from .label_smooth_cross_entropy_loss import CrossEntropyLabelSmooth
from .triplet_loss_with_batchhard import TripletLoss
from .criterion import Criterion


def build_criterion(cfg):

    __criterion_factory = {
        # original support
        'cross_entropy': torch.nn.CrossEntropyLoss,
        'cross_entropy_label_smooth': CrossEntropyLabelSmooth,
        'tripletloss': TripletLoss,
        'l2': torch.nn.MSELoss,
        # user customized
        **LOSSes_REGISTRY._obj_map
    }

    cfg = edict(cfg)
    criterion_list = []
    for key in cfg.keys():
        if 'loss' not in key: continue
        tmp = {}
        val = getattr(cfg, key)
        name = val.criterion.pop('name')
        try:
            tmp['criterion'] = __criterion_factory[name](num_classes=cfg.num_classes, **val.criterion)
        except:
            tmp['criterion'] = __criterion_factory[name](**val.criterion)
        tmp['inputs'] = val.inputs
        tmp['weight'] = val.weight
        tmp['display_name'] = val.display_name
        criterion_list.append(tmp)

    return Criterion(criterion_list)
