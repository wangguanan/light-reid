"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import numpy as np
import torch


class CatMeter:
    '''
    Concatenate Meter for torch.Tensor
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None

    def update(self, val):
        if self.val is None:
            self.val = val
        else:
            self.val = torch.cat([self.val, val], dim=0)
    def get_val(self):
        return self.val

    def get_val_numpy(self):
        return self.val.data.cpu().numpy()


class MultiItemAverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.content = {}

    def update(self, val):
        '''
        :param val: dict, keys are strs, values are torch.Tensor or np.array
        '''
        for key in list(val.keys()):
            value = val[key]
            if key not in list(self.content.keys()):
                self.content[key] = {'avg': value, 'sum': value, 'count': 1.0}
            else:
                self.content[key]['sum'] += value
                self.content[key]['count'] += 1.0
                self.content[key]['avg'] = self.content[key]['sum'] / self.content[key]['count']

    def get_val(self):
        keys = list(self.content.keys())
        values = []
        for key in keys:
            val = self.content[key]['avg']
            if isinstance(val, torch.Tensor):
                val = val.data.cpu().numpy()
            values.append(val)
        return keys, values

    def get_str(self):

        result = ''
        keys, values = self.get_val()

        for key, value in zip(keys, values):
            result += key
            result += ': '
            if isinstance(value, np.ndarray):
                value = np.round(value, 5)
            result += str(value)
            result += '; '

        return result


class AverageMeter:
    """
    Average Meter
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1

    def get_val(self):
        return self.sum / self.count
