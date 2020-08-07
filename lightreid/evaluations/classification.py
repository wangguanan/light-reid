"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import numpy as np

__all__ = ['accuracy']

def accuracy4tensor(output, target, topk=[1]):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return np.array(res)

def accuracy4list(output_list, target, topk=[1]):
    res = 0
    for output in output_list:
        res += 1/len(output_list) * accuracy4tensor(output, target, topk)
    return res

def accuracy(output, target, topk=[1]):
    """Computes the precision@k for the specified values of k"""
    if isinstance(output, list):
        return accuracy4list(output, target, topk)
    else:
        return accuracy4tensor(output, target, topk)
