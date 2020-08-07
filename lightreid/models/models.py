"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import torch
import torch.nn as nn
import copy


class BaseReIDModel(nn.Module):
    """
    Archtecture for Based ReID Model
    combine backbone, pooling and head modules
    """

    def __init__(self, backbone, pooling, head):
        super(BaseReIDModel, self).__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.head = head
        self.use_tanh = False

    def forward(self, x, y=None, teacher_mode=False, fixcnn=False, test_feat_from_head=True):
        '''
        Args:
            x(torch.tensor): input images
            y(torch.tensor): input labels, used for such as circle loss
        '''
        # cnn backbone
        feat_map = self.backbone(x)
        if fixcnn:
            feat_map = feat_map.detach()
        # pooling
        feat_vec = self.pooling(feat_map).squeeze(3).squeeze(2)

        # teacher mode
        if teacher_mode:
            _, logits = self.head(feat_vec, y, use_tanh=self.use_tanh, teacher_mode=True)
            return feat_vec, logits

        # return
        if self.training:
            _, logits = self.head(feat_vec, y, use_tanh=self.use_tanh)
            return feat_vec, logits
        else:
            if test_feat_from_head:
                bnfeat_vec = self.head(feat_vec, y, use_tanh=self.use_tanh)
                return bnfeat_vec
            else:
                return feat_vec

    def enable_tanh(self):
        self.use_tanh = True

    def disable_tanh(self):
        self.use_tanh = False


class PCBReIDModel(BaseReIDModel):

    def forward(self, x, y=None, fixcnn=False):
        # conn backbone
        feat_map = self.backbone(x)
        if fixcnn:
            feat_map = feat_map.detach()
        # pooling
        feat_vec = self.pooling(feat_map).squeeze(3)
        # return
        if self.training:
            embedding_list, logits_list = self.head(feat_vec, y)
            return embedding_list, logits_list
        else:
            feat = self.head(feat_vec, y)
            return feat


class TeacherReIDModel(BaseReIDModel):
    """
    Architecture for Teacher ReID Model
    return feat_vec and logits under training mode
    """

    def forward(self, x, y=None, enable_tanh=False):
        # conn backbone
        feat_map = self.backbone(x)
        # pooling
        feat_vec = self.pooling(feat_map).squeeze(3).squeeze(2)
        # return
        bnfeat_vec, logits = self.head(feat_vec, y, enable_tanh=enable_tanh, teacher_mode=True)
        return feat_vec, logits
