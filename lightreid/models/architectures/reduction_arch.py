"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import torch
import torch.nn as nn
import copy
import numpy as np

from .base_arch import BaseReIDModel
from sklearn.decomposition import PCA

from .build import ARCHs_REGISTRY

@ARCHs_REGISTRY.register()
class ReductionReIDModel(BaseReIDModel):
    """
    Architecture for ReID Model
    combine backbone, pooling and head modules
    reduce feature with pca
    """

    def __init__(self, backbone, pooling, head, reduction_dim, reduction_type='pca'):
        super(ReductionReIDModel, self).__init__(backbone, pooling, head)
        self.register_buffer('is_reduction', torch.tensor(1).to(torch.bool))
        self.queue = Queue(max_len=5000)
        self.alert = Alerter(max_iters=500)
        if reduction_type == 'pca':
            self.pca = PCA(n_components=reduction_dim)
            self.pca_mean_ = nn.Parameter(torch.rand(self.backbone.dim), requires_grad=False)
            self.pca_components_ = nn.Parameter(torch.rand(reduction_dim, self.backbone.dim), requires_grad=False)
        else:
            assert 0, 'reduction type error'
        self.enable_reduction()

    def reduce_fit(self, feats):
        '''
        compute mean(numpy) and components(numpy) with sklearn.pca
        and assign them to torch.nn.Parameter
        Args:
            feats(numpy.array): size [n_samples, features_dim]
        '''
        self.pca.fit(feats)
        device = torch.device('cuda') if self.pca_mean_.is_cuda else torch.device('cpu')
        self.pca_mean_.data = torch.from_numpy(self.pca.mean_).to(device).data
        self.pca_components_.data = torch.from_numpy(self.pca.components_).to(device).data

    def forward(self, x, y=None, fixcnn=False):
        '''
        if training mode or not reduce, return original resutls
        else (if eval mode and reduce), reduce feature with pca (zero-center and transform with self.pca_components_)
        '''
        # forward
        res = super(ReductionReIDModel, self).forward(x, y=y, fixcnn=fixcnn)
        # pca
        if self.training:
            self.queue.update(res[self.test_feats].data.cpu().numpy()) # record history features
            if self.alert.step(): self.reduce_fit(self.queue.get_val()) # pca
        # return
        if self.training or not self.is_reduction:
            return res
        else: # reduce dimension
            return torch.matmul(res - self.pca_mean_, self.pca_components_.t())

    def enable_reduction(self):
        device = torch.device('cuda') if self.is_reduction.is_cuda else torch.device('cpu')
        self.is_reduction.data = torch.tensor(1).to(device).data

    def disable_reduction(self):
        device = torch.device('cuda') if self.is_reduction.is_cuda else torch.device('cpu')
        self.is_reduction.data = torch.tensor(0).to(device).data


class Queue:

    def __init__(self, max_len):
        self.max_len = max_len
        self.init()

    def init(self):
        self.queue = None

    def update(self, tensor):
        '''
        Args:
            tensor(torch.tensor)
        '''
        if self.queue is None:
            self.queue = tensor
        else:
            self.queue = np.concatenate([self.queue, tensor], axis=0)
        if self.queue.shape[0] > self.max_len:
            len = self.queue.shape[0]
            self.queue = self.queue[len-self.max_len:]

    def get_val(self):
        return self.queue


class Alerter:
    '''
    return True every max_iters
    '''

    def __init__(self, max_iters):
        self.max_iters = max_iters
        self.count = 0

    def step(self):
        self.count += 1
        if self.count > self.max_iters:
            self.count = 0
            return True
        else:
            return False

