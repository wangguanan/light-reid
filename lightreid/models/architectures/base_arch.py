"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import torch
import torch.nn as nn
import copy

from .build import ARCHs_REGISTRY

__all__ = ['BaseReIDModel']

@ARCHs_REGISTRY.register()
class BaseReIDModel(nn.Module):
    """
    Architecture for ReID Model
    combine backbone, pooling and head modules
    """

    def __init__(self, backbone, pooling, head):
        super(BaseReIDModel, self).__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.head = head
        self.disable_hash()

    def forward(self, x, y=None, fixcnn=False):
        '''
        Args:
            x(torch.tensor): images
            y(torch.tensor): labels, required by circle_softmax, arc_softmax
        '''
        # cnn backbone
        feats_map = self.backbone(x)
        if fixcnn:
            feats_map = feats_map.detach()
        # pooling
        feats_vec = self.pooling(feats_map).squeeze(3).squeeze(2) # support bs=1
        # head
        res = self.head(feats_vec, y)

        # return
        res['feats_map'] = feats_map
        res['feats_vec'] = feats_vec

        if self.training:
            return res
        else:
            return res[self.test_feats]

    def enable_hash(self):
        self.head.enable_hash()
        self.test_feats = 'binary_feats'

    def disable_hash(self):
        self.head.disable_hash()
        self.test_feats = 'bn_feats'