"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import numpy as np
import os
from sklearn import metrics as sk_metrics
import matplotlib.pyplot as plt

from ..build import EVALUATORs_REGISTRY

__all__ = ['CmcMapEvaluator']

class BaseEvaluator:

    def evaluate(self):
        return NotImplementedError

    def cosine_dist(self, x, y):
        '''compute cosine distance between two martrix x and y with sizes (n1, d), (n2, d)'''
        return sk_metrics.pairwise.cosine_distances(x, y)

    def euclidean_dist(self, x, y):
        '''compute eculidean distance between two martrix x and y with sizes (n1, d), (n2, d)'''
        return sk_metrics.pairwise.euclidean_distances(x, y)

    def hamming_dist(self, x, y):
        assert x.shape[1] == y.shape[1]
        code_len = x.shape[1]
        if x.min() == 0: #{0,1} --> {-1,1}
            x = (x - 0.5)*2
            y = (y - 0.5)*2
        return code_len - (np.matmul(x, y.transpose([1,0])) + code_len) /2.0
        # return sk_metrics.pairwise_distances(x, y, metric=scipy_dist.hamming)


@EVALUATORs_REGISTRY.register()
class CmcMapEvaluator(BaseEvaluator):
    '''
    Compute Rank@k and mean Average Precision (mAP) scores for ReID task.
    Evaluate all query at one time.
    This is very fast due the parallel matrix computation of numpy. (computing distance and sort)
    But it takes much more memory.
    Args:
        metric(str): could be cosine, euclidean and hamming
        mode(str): could be inter-camera, intra-camera and all
    '''

    def __init__(self, metric, mode):
        assert metric in ['cosine', 'euclidean', 'hamming'], 'expect metric in cosine/euclidean/hamming, but got {}'.format(metric)
        assert mode in ['inter-camera', 'intra-camera', 'all']
        self.mode = mode
        self.metric = metric
        print(self.metric, self.mode)

    def evaluate(self, query_features, query_camids, query_pids, gallery_features, gallery_camids, gallery_pids):
        '''
        query_features(np.ndarray): [sample_num, feat_dim]
        query_camids(np.ndarray): [sample_num]
        query_pids(np.ndarray): [sample_num]
        gallery_features(np.ndarray): [sample_num, feat_dim]
        gallery_camids(np.ndarray): [sample_num]
        gallery_pids(np.ndarray): [sample_num]
        '''

        '''compute distance matrix'''
        if self.metric == 'cosine':
            scores = self.cosine_dist(query_features, gallery_features)
        elif self.metric == 'euclidean':
            scores = self.euclidean_dist(query_features, gallery_features)
        elif self.metric == 'hamming':
            scores = self.hamming_dist(query_features, gallery_features)
        else:
            assert 0, 'metric error, got {}.'.format(self.metric)
        rank_results = np.argsort(scores)

        '''evaluate every query'''
        APs, CMC = [], []
        for idx, data in enumerate(zip(rank_results, query_camids, query_pids)):
            a_rank, query_camid, query_pid = data
            ap, cmc = self.compute_AP(a_rank, query_camid, query_pid, gallery_camids, gallery_pids)
            APs.append(ap), CMC.append(cmc)

        '''compute CMC and mAP'''
        MAP = np.array(APs).mean()
        min_len = min([len(cmc) for cmc in CMC])
        CMC = [cmc[:min_len] for cmc in CMC]
        CMC = np.mean(np.array(CMC), axis=0)

        return MAP, CMC


    def compute_AP(self, a_rank, query_camid, query_pid, gallery_camids, gallery_pids):
        '''given a query and all galleries, compute its ap and cmc'''

        if self.mode == 'inter-camera':
            junk_index_1 = self.in1d(np.argwhere(query_pid == gallery_pids), np.argwhere(query_camid == gallery_camids))
            junk_index_2 = np.argwhere(gallery_pids == -1)
            junk_index = np.append(junk_index_1, junk_index_2)
            index_wo_junk = self.notin1d(a_rank, junk_index)
            good_index = self.in1d(np.argwhere(query_pid == gallery_pids), np.argwhere(query_camid != gallery_camids))
        elif self.mode == 'intra-camera':
            junk_index_1 = np.argwhere(query_camid != gallery_camids)
            junk_index_2 = np.argwhere(gallery_pids == -1)
            junk_index = np.append(junk_index_1, junk_index_2)
            index_wo_junk = self.notin1d(a_rank, junk_index)
            good_index = np.argwhere(query_pid == gallery_pids)
            # self_junk = a_rank[0] if a_rank[0] == 1 or a_rank[0] == 0 else []# remove self
            self_junk = a_rank[0]
            index_wo_junk = np.delete(index_wo_junk, np.where(self_junk == index_wo_junk))
            good_index = np.delete(good_index, np.where(self_junk == good_index))
        elif self.mode == 'all':
            junk_index = np.argwhere(gallery_pids == -1)
            index_wo_junk = self.notin1d(a_rank, junk_index)
            good_index = np.argwhere(query_pid == gallery_pids)
            self_junk = a_rank[0] if a_rank[0] == 1 or a_rank[0] == 0 else [] # remove self if euclidean distance == 0 or cosine similarity == 1
            index_wo_junk = np.delete(index_wo_junk, np.where(self_junk == index_wo_junk))
            good_index = np.delete(good_index, np.where(self_junk == good_index))

        # num_good = len(good_index)
        hit = np.in1d(index_wo_junk, good_index)
        index_hit = np.argwhere(hit == True).flatten()
        if len(index_hit) == 0:
            AP = 0
            cmc = np.zeros([len(index_wo_junk)])
        else:
            precision = []
            for i in range(len(index_hit)):
                precision.append(float(i + 1) / float((index_hit[i] + 1)))
            AP = np.mean(np.array(precision))
            cmc = np.zeros([len(index_wo_junk)])
            cmc[index_hit[0]:] = 1
        return AP, cmc

    def in1d(self, array1, array2, invert=False):
        '''
        :param set1: np.array, 1d
        :param set2: np.array, 1d
        :return:
        '''
        mask = np.in1d(array1, array2, invert=invert)
        return array1[mask]

    def notin1d(self, array1, array2):
        return self.in1d(array1, array2, invert=True)


@EVALUATORs_REGISTRY.register()
class PreRecEvaluator(BaseEvaluator):
    '''
    Compute Precision and Recall for Re-ID task
    '''

    def __init__(self, metric, mode):
        assert metric in ['cosine', 'euclidean']
        assert mode in ['intra-camera', 'inter-camera', 'all']
        self.metric = metric
        self.mode = mode

    def evaluate(self, query_features, query_camids, query_pids, gallery_features, gallery_camids, gallery_pids):
        '''
        Args:
            thresholds(list):
            query_features(np.ndarray): [sample_num, feat_dim]
            query_camids(np.ndarray): [sample_num]
            query_pids(np.ndarray): [sample_num]
            gallery_features(np.ndarray): [sample_num, feat_dim]
            gallery_camids(np.ndarray): [sample_num]
            gallery_pids(np.ndarray): [sample_num]
        '''

        thresholds = np.linspace(1.0, 0.0, num=21)

        '''compute distance matrix'''
        if self.metric is 'cosine':
            scores = self.cosine_dist(query_features, gallery_features)
        elif self.metric is 'euclidean':
            scores = self.euclidean_dist(query_features, gallery_features)
        else:
            assert 0, 'dist type error'

        pid_similarity = (np.expand_dims(query_pids, axis=0).transpose([1,0]) == np.expand_dims(gallery_pids, axis=0)).astype(np.float)
        cid_similarity = (np.expand_dims(query_camids, axis=0).transpose([1,0]) == np.expand_dims(gallery_camids, axis=0)).astype(np.float)

        pres, recalls = [], []
        for threshold in thresholds:
            if self.metric == 'cosine':
                hits = scores >= threshold
            elif self.metric == 'euclidean':
                hits = scores <= threshold
            else:
                assert 0, 'dist type error'

            if self.mode == 'all':
                pre = (pid_similarity * hits).sum() / hits.sum()
                recall = (pid_similarity * hits).sum() / pid_similarity.sum()
            elif self.mode == 'intra-camera':
                pre = (pid_similarity * cid_similarity * hits).sum() / (cid_similarity * hits).sum()
                recall = (pid_similarity * cid_similarity * hits).sum() / (pid_similarity * cid_similarity).sum()
            elif self.mode == 'inter-camera':
                pre = (pid_similarity * (1-cid_similarity) * hits).sum() / ((1-cid_similarity) * hits).sum()
                recall = (pid_similarity * (1-cid_similarity) * hits).sum() / (pid_similarity * (1-cid_similarity)).sum()
            else:
                assert 0, 'mode type error'

            pres.append(pre)
            recalls.append(recall)

        return pres, recalls, thresholds

    def plot_prerecall_curve(self, path, pres, recalls, mAP=None, CMC=None, label=None):

        if mAP is not None and CMC is not None and label is not None:
            plt.plot(recalls, pres, label='{model},map:{map},cmc135:{cmc}'.format(
                model=label, map=round(mAP, 2), cmc=[round(CMC[0], 2), round(CMC[2], 2), round(CMC[4], 2)]))
        else:
            plt.plot(recalls, pres)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('precision-recall curve')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(path, 'precisio-recall-curve.png'))




