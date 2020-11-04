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

    def __init__(self, in_dim, class_num, classifier=None, middle_dim=None, normalize_feats=True):
        super(BNHead, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num
        self.normalize_feats = normalize_feats

        self.bn = nn.BatchNorm2d(self.in_dim)
        self.bn.bias.requires_grad_(False)

        self.middle_dim = middle_dim
        if middle_dim is not None:
            self.middle_fc = nn.Linear(in_dim, middle_dim)
            self.middle_bn = nn.BatchNorm1d(middle_dim)
            self.middle_bn.bias.requires_grad_(False)

        if classifier is None:
            tmp_tim = self.in_dim if middle_dim is None else middle_dim
            self.classifier = nn.Linear(tmp_tim, self.class_num, bias=False)
        else:
            self.classifier = classifier

        self.bn.apply(weights_init_kaiming)
        if middle_dim is not None:
            self.middle_fc.apply(weights_init_classifier)
            self.middle_bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)


    def forward(self, feats, y=None, use_tanh=False, teacher_mode=False):
        bn_feats = self.bn(feats)

        if self.middle_dim is not None:
            feats = self.middle_fc(bn_feats)
            bn_feats = self.middle_bn(feats)

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
                if self.normalize_feats:
                    return torch.nn.functional.normalize(bn_feats, dim=1, p=2) # real-value feats
                else:
                    return bn_feats

        # train
        if self.classifier.__class__.__name__ in ['Circle', 'ArcFace']:
            logits = self.classifier(bn_feats, y)
        else:
            logits = self.classifier(bn_feats)
        logits2 = F.linear(bn_feats, self.classifier.weight)
        return bn_feats, (logits, logits2)
