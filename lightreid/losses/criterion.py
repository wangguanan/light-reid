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
        tmp['inputs'] = val.inputs
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


    def __init__(self, criterion_list):
        self.criterion_list = criterion_list

    def compute(self, **kwargs):

        overall_loss = 0
        loss_dict = {}

        # compute weighted loss
        for value in self.criterion_list:
            weight = value['weight']
            criterion = value['criterion']
            inputs = value['inputs']
            if isinstance(inputs, str):
                loss = weight * criterion(kwargs[inputs])
            elif isinstance(inputs, dict):
                inputs_tmp = {}
                for key, val in inputs.items():
                    inputs_tmp[key] = kwargs[val]
                loss = weight * criterion(**inputs_tmp)
            else:
                raise RuntimeError('type error')

            overall_loss += loss
            name = criterion.__class__.__name__ if 'display_name' not in value.keys() else value['display_name']
            loss_dict[name] = loss.data
            del loss
            value, weight, criterion, inputs = None, None, None, None

        return overall_loss, loss_dict

