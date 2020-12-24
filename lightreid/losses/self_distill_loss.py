"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import LOSSes_REGISTRY

@LOSSes_REGISTRY.register()
class ProbSelfDistillLoss:
    '''
    Probability Self Distillation Loss
    Reference:
    Paper: Wang et al. Faster Person Re-Identification. ECCV2020

    Args:
        logits_list(list): list of logits, the first one is teacher, all the others are students
    '''

    def __init__(self):
        super(ProbSelfDistillLoss, self).__init__()

    def __call__(self, logits_list):
        s_logits_list = logits_list[1:]
        t_logits_list = [logits_list[0].detach() for _ in range(len(s_logits_list))]
        return self.prob_distill_loss(s_logits_list, t_logits_list)

    def prob_distill_loss(self, s_logits_list, t_logits_list):

        def kl_div_loss(logits_s, logits_t, mini=1e-8):
            '''
            :param logits_s: student score
            :param logits_t: teacher score as target
            :param mini: for number stable
            :return:
            '''
            logits_t = logits_t.detach()
            prob1 = F.softmax(logits_s, dim=1)
            prob2 = F.softmax(logits_t, dim=1)
            loss = torch.sum(prob2 * torch.log(mini + prob2 / (prob1 + mini)), 1) + \
                   torch.sum(prob1 * torch.log(mini + prob1 / (prob2 + mini)), 1)
            return loss.mean()

        loss = 0
        for s_logits, t_logits in zip(s_logits_list, t_logits_list):
            loss += kl_div_loss(s_logits, t_logits)
        return loss


class SIMSelfDistillLoss:
    '''
    Self-Distillation Loss of Similarity

    Reference:
    Paper: Wang et al. Faster Person Re-Identification. ECCV2020

    Args:
        feats_list(list): a list of feats. the first one is teacher, all the others are students
    '''

    def __init__(self):
        super(SIMSelfDistillLoss, self).__init__()

    def __call__(self, feats_list):
        s_feats_list = feats_list[1:]
        t_feats_list = [feats_list[0].detach() for _ in range(len(s_feats_list))]
        return self.sim_distill_loss(s_feats_list, t_feats_list)

    def sim_distill_loss(self, s_feats_list, t_feats_list):
        '''
        compute similarity distillation loss
        :param score_list:
        :param mimic(list): [teacher, student]
        :return:
        '''
        loss = 0
        for s_feats, t_feats in zip(s_feats_list, t_feats_list):
            s_similarity = torch.mm(s_feats, s_feats.transpose(0, 1))
            s_similarity = F.normalize(s_similarity, p=2, dim=1)
            t_similarity = torch.mm(t_feats, t_feats.transpose(0, 1)).detach()
            t_similarity = F.normalize(t_similarity, p=2, dim=1)
            loss += (s_similarity - t_similarity).pow(2).mean()
        return loss
