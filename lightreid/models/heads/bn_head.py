"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightreid.utils import weights_init_kaiming, weights_init_classifier

class BNHead(nn.Module):
    """
    features --> bn --> tanh(optional) --> fc --> logits
    """

    def __init__(self, in_dim, class_num, classifier=None):
        super(BNHead, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        if classifier is None:
            self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)
        else:
            self.classifier = classifier

        self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, feats, y=None, use_tanh=False, teacher_mode=False):
        bn_feats = self.bn(feats)
        if use_tanh:
            tanh_feats = torch.tanh(bn_feats)
            bn_feats = tanh_feats

        # teacher mode: return bn_feats and logits
        if teacher_mode:
            if self.classifier.__class__.__name__ in ['Circle', 'ArcFace']:
                logits = self.classifier(bn_feats, y)
            else:
                logits = self.classifier(bn_feats)
            logits2 = F.linear(bn_feats, self.classifier.weight)
            return bn_feats, (logits, logits2)

        # eval
        if not self.training:
            if use_tanh:
                return (torch.sign(bn_feats) + 1.0)/2.0 # binary codes, i.e. {0,1}
            else:
                return bn_feats # real-value feats

        # train
        if self.classifier.__class__.__name__ in ['Circle', 'ArcFace']:
            logits = self.classifier(bn_feats, y)
        else:
            logits = self.classifier(bn_feats)
        logits2 = F.linear(bn_feats, self.classifier.weight)
        return bn_feats,  (logits, logits2)
