"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import LOSSes_REGISTRY

__all__ = ['TripletLoss']

@LOSSes_REGISTRY.register()
class RankingLoss:

    def __init__(self):
        pass

    def _label2similarity(sekf, label1, label2):
        '''
        compute similarity matrix of label1 and label2
        :param label1: torch.Tensor, [m]
        :param label2: torch.Tensor, [n]
        :return: torch.Tensor, [m, n], {0, 1}
        '''
        m, n = len(label1), len(label2)
        l1 = label1.view(m, 1).expand([m, n])
        l2 = label2.view(n, 1).expand([n, m]).t()
        similarity = l1 == l2
        return similarity

    def _batch_hard(self, mat_distance, mat_similarity, more_similar):

        if more_similar is 'smaller':
            sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
            hard_p = sorted_mat_distance[:, 0]
            sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
            hard_n = sorted_mat_distance[:, 0]
            return hard_p, hard_n

        elif more_similar is 'larger':
            sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1, descending=False)
            hard_p = sorted_mat_distance[:, 0]
            sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1, descending=True)
            hard_n = sorted_mat_distance[:, 0]
            return hard_p, hard_n


class TripletLoss(RankingLoss):
    '''
    Compute Triplet loss augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    Args:
        margin(float or 'soft'): if float, use nn.MarginRankingLoss, if 'soft',
    '''

    def __init__(self, margin, metric, reduce=True):
        '''
        :param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
        :param bh: batch hard
        :param metric: l2 distance or cosine distance
        '''

        assert isinstance(margin, float) or margin=='soft', \
            'margin must be type float or value \'soft\', but got {}'.format(margin)
        if isinstance(margin, float):
            self.margin_loss = nn.MarginRankingLoss(margin=margin, reduce=reduce)
        elif margin == 'soft':
            self.margin_loss = nn.SoftMarginLoss(reduce=reduce)
        self.metric = metric

    def __call__(self, emb, label):
        '''
        :param emb: torch.Tensor, [m, dim]
        :param label: torch.Tensor, [b]
        '''

        if self.metric == 'cosine':
            mat_dist = cosine_dist(emb, emb)
            mat_sim = self._label2similarity(label, label)
            hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

            mat_dist = cosine_dist(emb, emb)
            mat_sim = self._label2similarity(label, label)
            _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

            margin_label = -torch.ones_like(hard_p)

        elif self.metric == 'euclidean':
            mat_dist = euclidean_dist(emb, emb)
            mat_sim = self._label2similarity(label, label)
            hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

            mat_dist = euclidean_dist(emb, emb)
            mat_sim = self._label2similarity(label, label)
            _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

            margin_label = torch.ones_like(hard_p)

        if self.margin_loss.__class__.__name__ == 'MarginRankingLoss':
            return self.margin_loss(hard_n, hard_p, margin_label)
        elif self.margin_loss.__class__.__name__ == 'SoftMarginLoss':
            return self.margin_loss(hard_n-hard_p, margin_label)


def cosine_dist(x, y):
    '''
    compute cosine distance between two matrix x and y
    with size (n1, d) and (n2, d) and type torch.tensor
    return a matrix (n1, n2)
    '''

    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return torch.matmul(x, y.transpose(0, 1))


def euclidean_dist(x, y):
    """
    compute euclidean distance between two matrix x and y
    with size (n1, d) and (n2, d) and type torch.tensor
    return a matrix (n1, n2)
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist
