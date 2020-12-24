"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import torch.nn.functional as F
from .build import LOSSes_REGISTRY

@LOSSes_REGISTRY.register()
class KLLoss:
    '''KL Divergence'''

    def __init__(self, t=4):
        self.t = t

    def __call__(self, logits_s, logits_t):
        p_s = F.log_softmax(logits_s / self.t, dim=1)
        p_t = F.softmax(logits_t / self.t, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.t ** 2) / logits_s.shape[0]
        return loss