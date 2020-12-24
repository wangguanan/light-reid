"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import torch
import torch.nn as nn

from .bn_head import BNHead
import lightreid

from .build import HEADs_REGISTRY


@HEADs_REGISTRY.register()
class CodePyramid(nn.Module):
    """Pyramid Head.
    Learn multiple codes of different lengths.

    Reference:
    Paper: Wang et al. Faster Person Re-identification. ECCV 2020.

    Args:
        in_dim (int): input feature dimension
        out_dims (list): out feature dimensions, e.g. out_dims=[2048, 512, 128, 32]
    """

    def __init__(self, in_dim, out_dims, class_num, head='BNHead', classifier='Linear'):
        super(CodePyramid, self).__init__()
        self.in_dim = in_dim
        self.eval_dims = out_dims
        self.class_num = class_num
        self.train_dims = [2048, 1024, 512, 256, 128, 64, 32]

        assert head in ['BNHead'], 'expect head in [\'BNHead\'] but got {}'.format(head)
        assert classifier in ['Linear', 'Circle'], 'expect classifier in [\'Linear\', \'Circle\'], but got {}'.format(classifier)

        setattr(self, 'neck{}'.format(int(in_dim)), BNHead(int(in_dim), self.class_num))
        for idx, dim in enumerate(self.train_dims):
            if idx == 0:
                setattr(self, 'fc{}'.format(int(dim)), nn.Linear(in_dim, int(dim)))
            else:
                setattr(self, 'fc{}'.format(int(dim)), nn.Linear(int(self.train_dims[idx - 1]), int(dim)))
            if classifier == 'Circle':
                neck = BNHead(int(dim), self.class_num, lightreid.models.Circle(in_dim, self.class_num, scale=64, margin=0.35))
            elif classifier == 'Linear':
                neck = BNHead(int(dim), self.class_num)
            setattr(self, 'neck{}'.format(int(dim)), neck)

    def forward(self, feats, y=None, use_tanh=False):

        neck = getattr(self, 'neck{}'.format(int(self.in_dim)))

        if self.training:
            bn_feats, logits = neck(feats, use_tanh=use_tanh)
            feats_list = [feats]
            bnfeats_list = [bn_feats]
            logits_list = [[logits[0]], [logits[1]]]
        else:
            binary_codes = neck(feats, use_tanh=use_tanh)
            binary_codes_list = [binary_codes]

        for idx, dim in enumerate(self.train_dims):
            fc = getattr(self, 'fc{}'.format(int(dim)))
            neck = getattr(self, 'neck{}'.format(int(dim)))
            feats = fc(feats.detach())

            if self.training:
                bn_feats, logits = neck(feats, use_tanh=use_tanh)
                feats_list.append(feats)
                bnfeats_list.append(bn_feats)
                logits_list[0].append(logits[0])
                logits_list[1].append(logits[1])
            else:
                binary_codes = neck(feats, use_tanh=use_tanh)
                binary_codes_list.append(binary_codes)

        if self.training:
            return feats_list, bnfeats_list, logits_list
        else:
            binary_codes_list = binary_codes_list[1:]
            return [val for val in binary_codes_list if val.shape[1] in self.eval_dims]
