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
        feats_map = self.backbone(x)
        if fixcnn:
            feats_map = feats_map.detach()
        # pooling
        feats_vec = self.pooling(feats_map).squeeze(3).squeeze(2)

        # teacher mode
        if teacher_mode:
            headfeats_vec, logits = self.head(feats_vec, y, use_tanh=self.use_tanh, teacher_mode=True)
            return feats_vec, headfeats_vec, logits

        # return
        if self.training:
            if 'BNHead' in self.head.__class__.__name__ :
                headfeats_vec, logits = self.head(feats_vec, y, use_tanh=self.use_tanh)
                return feats_vec, headfeats_vec, logits
            elif self.head.__class__.__name__ == 'CodePyramid':
                feats_list, headfeats_list, logits_list = self.head(feats_vec, y, use_tanh=self.use_tanh)
                return feats_list, headfeats_list, logits_list
            else:
                assert 0, 'head error, got {}'.format(self.head.__class__.__name__)
        else:
            if test_feat_from_head:
                headfeats_vec = self.head(feats_vec, y, use_tanh=self.use_tanh)
                return headfeats_vec
            else:
                return feats_vec

    def enable_tanh(self):
        self.use_tanh = True

    def disable_tanh(self):
        self.use_tanh = False


# class PCBReIDModel(BaseReIDModel):
#
#     def forward(self, x, y=None, fixcnn=False):
#         # conn backbone
#         feat_map = self.backbone(x)
#         if fixcnn:
#             feat_map = feat_map.detach()
#         # pooling
#         feat_vec = self.pooling(feat_map).squeeze(3)
#         # return
#         if self.training:
#             embedding_list, logits_list = self.head(feat_vec, y)
#             return embedding_list, logits_list
#         else:
#             feat = self.head(feat_vec, y)
#             return feat
