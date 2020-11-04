"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

from easydict import EasyDict as edict
import torch

from .label_smooth_cross_entropy_loss import CrossEntropyLabelSmooth
from .triplet_loss_with_batchhard import TripletLoss

__criterion_factory = {
    'cross_entropy': torch.nn.CrossEntropyLoss,
    'cross_entropy_label_smooth': CrossEntropyLabelSmooth,
    'tripletloss': TripletLoss,

}

def build_criterion(cfg):

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
        tmp['input'] = val.input
        tmp['weight'] = val.weight
        criterion_list.append(tmp)

    return Criterion(criterion_list)


class Criterion(object):
    '''
    Example:
        criterion = Creterion([
            {'criterion': torch.nn.CrossEntropyLoss(), 'weight': 1.0},
            {'criterion': lightreid.losses.TripletLoss(margin=0.3, metric='cosine'), 'weight': 1.0},
        ])

        imgs, pids, camids = data
        feats, logits = model(imgs)
        loss = criterion.compute(feats=feats, logits=logits, pids=pids)
    '''

    CRITERION_FACTORY = [
        'CrossEntropyLoss', 'CrossEntropyLabelSmooth', 'TripletLoss', 'FocalLoss', 'CenterLoss',
        'ProbSelfDistillLoss', 'SIMSelfDistillLoss', 'KLLoss']
    VALUE_FACTORY = [
        'feats', 'head_feats', 'logits', 'cls_score', 'pids', 'camids', # for vanilla traning
        'feats_s', 'logits_s', 'feats_t', 'logits_t', # distillation
        'reduce',
    ]

    def __init__(self, criterion_list):
        # check criterion class
        for criterion in criterion_list:
            assert criterion['criterion'].__class__.__name__ in Criterion.CRITERION_FACTORY, \
                'expect one of {}, but got {}'.format(Criterion.CRITERION_FACTORY, criterion['criterion'].__class__.__name__)
        self.criterion_list = criterion_list

    def compute(self, **kwargs):

        overall_loss = 0
        loss_dict = {}

        # check input is legal
        for arg in kwargs.keys():
            assert arg in Criterion.VALUE_FACTORY, \
                'expect one of {}, but got {}'.format(Criterion.VALUE_FACTORY, arg)

        # compute weighted loss
        for value in self.criterion_list:
            weight = value['weight']
            criterion = value['criterion']

            if criterion.__class__.__name__ in ['CrossEntropyLoss', 'CrossEntropyLabelSmooth', 'FocalLoss', 'CenterLoss']:
                inputs = value['input'] if 'input' in value.keys() else 'logits'
                if isinstance(kwargs[inputs], list): # for multi-head model (e.g. pcb, code pyramid), compute their sum/average loss
                    loss = 0
                    for idx in range(len(kwargs[inputs])):
                        loss += weight * criterion(kwargs[inputs][idx], kwargs['pids'])
                    if 'reduce' in value.keys() and value['reduce'] == 'mean':
                        loss *= 1 / len(kwargs[inputs])
                else: # for single-head model (e,g, bagtrick)
                    loss = weight * criterion(kwargs[inputs], kwargs['pids'])

            elif criterion.__class__.__name__ in ['TripletLoss']:
                inputs = value['input'] if 'input' in value.keys() else 'feats'
                if isinstance(kwargs[inputs], list): # for multi-head model (e.g. pcb), compute their average loss
                    loss = 0
                    for idx in range(len(kwargs[inputs])):
                        loss += weight * criterion(kwargs[inputs][idx], kwargs['pids'])
                    if 'reduce' in value.keys() and value['reduce'] == 'mean':
                        loss *= 1 / len(kwargs[inputs])
                else: # for single-head model (e,g, ide, bot)
                    loss = weight * criterion(kwargs[inputs], kwargs['pids'])

            elif criterion.__class__.__name__ in ['SIMSelfDistillLoss']:
                inputs = value['input'] if 'input' in value.keys() else 'head_feats'
                assert inputs in kwargs.keys(), \
                    'SimDistillLoss expect {} as inputs, but got {}'.format(inputs, kwargs.keys())
                assert isinstance(kwargs[inputs], list), \
                    'SIMSelfDistillLoss expect {} is type list, but got type {}'.format(inputs, type(kwargs[inputs]))
                loss = weight * criterion(feats_list=kwargs[inputs])

            elif criterion.__class__.__name__ in ['ProbSelfDistillLoss']:
                inputs = value['input'] if 'input' in value.keys() else 'logits'
                assert inputs in kwargs.keys(), \
                    'SimDistillLoss expect {} as inputs, but got {}'.format(inputs, kwargs.keys())
                loss = weight * criterion(logits_list=kwargs[inputs])

            elif criterion.__class__.__name__ in ['KLLoss']:
                assert 'logits_s' in kwargs.keys() and 'logits_t' in kwargs.keys(), \
                    'KLLoss expect logits_s and logits_t as inputs, but got {}'.format(kwargs.keys())
                if isinstance(kwargs['logits'], list):
                    loss = weight * criterion(logits_s=kwargs['logits_s'][0], logits_t=kwargs['logits_t'].detach())
                else:
                    loss = weight * criterion(logits_s=kwargs['logits_s'], logits_t=kwargs['logits_t'].detach())

            else:
                assert 0, 'expect criterion in {} but got {}'.format(Criterion.CRITERION_FACTORY, criterion)

            overall_loss += loss
            loss_dict[criterion.__class__.__name__] = loss.data
            del loss
            value, weight, criterion, inputs = None, None, None, None

        return overall_loss, loss_dict

