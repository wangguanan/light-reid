"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightreid.utils import weights_init_kaiming, weights_init_classifier
from lightreid.models.layers import build_classifier
from .build import HEADs_REGISTRY


@HEADs_REGISTRY.register()
class BNHead(nn.Module):
    """
    features + bn -->
    middle_fc + middle_bn (optional for feature reduction) -->
    tanh (optional for feature binarization) -->
    classifier_fc --> class_logits
    Args:
        in_dim(int): input feature dimentions, such as 2048
        class_num(int): class number to prediction
        classifier(dict): e.g. {'name': 'linear'}, {'name': 'circle', 'margin': 0.35, 'scale': 64}
        middle_dim(int): middle_dim for feature reduction with fc+bn, if None DO NOT reduction
    """

    def __init__(self, in_dim, class_num, classifier={'name': 'linear'}, middle_dim=None):
        super(BNHead, self).__init__()

        # parameters
        self.in_dim = in_dim
        self.class_num = class_num
        self.middle_dim = middle_dim
        # self.is_hash = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.register_buffer('is_hash', torch.tensor(0).to(torch.bool))

        # bn layer
        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)

        # feature reduction with fc+bn
        if middle_dim is not None:
            self.middle_fc = nn.Linear(in_dim, middle_dim)
            self.middle_bn = nn.BatchNorm1d(middle_dim)
            self.middle_bn.bias.requires_grad_(False)

        classifier['in_dim'] = in_dim if middle_dim is None else middle_dim
        classifier['out_dim'] = class_num
        self.classifier = build_classifier(**classifier)

        # initialize weights
        self.bn.apply(weights_init_kaiming)
        if middle_dim is not None:
            self.middle_fc.apply(weights_init_classifier)
            self.middle_bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, feats, y=None):
        """
        feats(tensor): batch features with size [bs, dim]
        y(tensor): labels with size [bs]
        """

        # bn layer
        bn_feats = self.bn(feats)

        # feature reduction
        if self.middle_dim is not None:
            feats = self.middle_fc(bn_feats)
            bn_feats = self.middle_bn(feats)

        # feature binarization
        if self.is_hash:
            tanh_feats = torch.tanh(bn_feats)
            binary_feats = (torch.sign(bn_feats) + 1.0) / 2.0  # binary codes, i.e. {0,1}

        # return in eval setting
        if not self.training:
            res = {
                'feats': feats,
                'bn_feats': bn_feats
            }
            if self.is_hash:
                res['tanh_feats'] = tanh_feats
                res['binary_feats'] = binary_feats
            return res

        # train, multiply classifier and output class logits
        feats_tmp = tanh_feats if self.is_hash else bn_feats
        if self.classifier.__class__.__name__ in ['Circle', 'ArcFace']:
            logits = self.classifier(feats_tmp, y)
        else:
            logits = self.classifier(feats_tmp)
        logits_distill = F.linear(feats_tmp, self.classifier.weight) # only used for distillation
        res = {
            'feats': feats,
            'bn_feats': bn_feats,
            'logits': logits,
            'logits_distill': logits_distill
        }
        if self.is_hash:
            res['tanh_feats'] = tanh_feats
            res['binary_feats'] = binary_feats
        return res


    def enable_hash(self):
        device = torch.device('cuda') if self.is_hash.is_cuda else torch.device('cpu')
        self.is_hash.data = torch.tensor(1).to(device).data

    def disable_hash(self):
        device = torch.device('cuda') if self.is_hash.is_cuda else torch.device('cpu')
        self.is_hash.data = torch.tensor(0).to(torch.bool).to(device).data
