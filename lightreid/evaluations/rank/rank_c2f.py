import numpy as np
import copy
from hexhamming import hamming_distance
import progressbar
import time

import lightreid
from lightreid.utils.meters import AverageMeter

import scipy
from scipy.optimize import curve_fit
from scipy import special
from scipy.optimize import minimize_scalar

from ..build import EVALUATORs_REGISTRY


__all__ = ['CmcMapEvaluatorC2F']


class FileTransferFrequency(progressbar.FileTransferSpeed):

    def _speed(self, value, elapsed):
        speed = elapsed / float(value)
        return speed * 1000, 1


@EVALUATORs_REGISTRY.register()
class CmcMapEvaluatorC2F:
    '''
    Evaluate every query one-by-one, including computing distance and sort.
    This is more practical in real-world Re-ID application.
    Speed up with binary code and Coarse-to-Fine search.
    Compute Rank@k and mean Average Precision (mAP) scores
    '''

    def __init__(self, metric, mode):

        assert metric in ['hamming'], 'expect hamming, but got {}'.format(metric)
        assert mode in ['inter-camera', 'intra-camera', 'all'], 'expect inter-camera/intra-camera and all, but got'.format(metric)
        self.metric = metric
        self.mode = mode

        # please kindly install progressbar library with command: ```pip install progressbar2```
        self.bar_rank = progressbar.ProgressBar(widgets=[
            'Ranking (Compute Hamming Dist and Counting Sort): ',
            progressbar.Percentage(),
            progressbar.Bar(),
            progressbar.SimpleProgress(format='%(value_s)s/%(max_value_s)s'),
            ' [', progressbar.Timer(), ',', FileTransferFrequency(format='%(scaled)5.1f ms/query'), '] '
        ])
        self.bar_evaluate = progressbar.ProgressBar(widgets=[
            'Evaluating (Compute mAP and CMC): ',
            progressbar.Percentage(),
            progressbar.Bar(),
            progressbar.SimpleProgress(format='%(value_s)s/%(max_value_s)s'),
            ' [', progressbar.Timer(), ',', FileTransferFrequency(format='%(scaled)5.1f ms/query'), '] '
        ])


    def compute(self, query_feats_list, query_camids, query_pids, gallery_feats_list, gallery_camids, gallery_pids, return_time=False):
        '''rank and evaluate'''

        query_feat_lens = [val.shape[1] for val in query_feats_list]
        gallery_feat_lens = [val.shape[1] for val in gallery_feats_list]
        assert query_feat_lens == gallery_feat_lens, \
            'query_feat_lens and gallery_feat_lens should be equal, but got {} and {}'.format(
                query_feat_lens, gallery_feat_lens)
        query_feats_list = [query_feats_list[idx] for idx in np.argsort(query_feat_lens)]
        gallery_feats_list = [gallery_feats_list[idx] for idx in np.argsort(query_feat_lens)]

        # compute threshold
        # thresholds = {32: 13, 128: 57, 512: 495, 2048: 483}
        thresholds = ThresholdOptimization(beta=2).optimize(query_feats_list, gallery_feats_list, query_pids, gallery_pids)
        print(thresholds)

        if self.metric == 'hamming': # convert np.ndarray to hex
            query_hex_list = []
            gallery_hex_list = []
            for query_feats, gallery_feats in zip(query_feats_list, gallery_feats_list):
                query_hex, gallery_hex = [], []
                code_len = query_feats.shape[1]
                for query_feat in query_feats:
                    binary_str = ''.join(str(int(i)) for i in query_feat)
                    hex_str = hex(int(binary_str, 2))[2:].zfill(int(code_len / 4))
                    query_hex.append(hex_str)
                for gallery_feat in gallery_feats:
                    binary_str = ''.join(str(int(i)) for i in gallery_feat)
                    hex_str = hex(int(binary_str, 2))[2:].zfill(int(code_len / 4))
                    gallery_hex.append(hex_str)
                query_hex_list.append(query_hex)
                gallery_hex_list.append(gallery_hex)
            query_feats_list = query_hex_list
            gallery_feats_list = gallery_hex_list

        # rank
        rank_time_meter = AverageMeter()
        all_rank_list = [[] for _ in range(len(query_pids))]
        for query_idx in self.bar_rank(range(len(query_pids))):
            ts = time.time()
            rank_list = self.rank_coarse2fine(query_idx, query_feats_list, gallery_feats_list, thresholds)
            rank_time_meter.update(time.time()-ts)
            all_rank_list[query_idx] = rank_list

        # evaluate
        eval_time_meter = AverageMeter()
        APs, CMC = [], []
        for query_idx in self.bar_evaluate(range(len(query_pids))):
            ts = time.time()
            AP, cmc = self.evaluate(
                query_idx, query_camids, query_pids,
                gallery_camids, gallery_pids, np.array(all_rank_list[query_idx]))
            eval_time_meter.update(time.time()-ts)
            # record
            APs.append(AP); CMC.append(cmc)

        # compute CMC and mAP
        MAP = np.array(APs).mean()
        min_len = min([len(cmc) for cmc in CMC])
        CMC = [cmc[:min_len] for cmc in CMC]
        CMC = np.mean(np.array(CMC), axis=0)

        if return_time:
            return MAP, CMC, rank_time_meter.get_val(), eval_time_meter.get_val()
        return MAP, CMC


    def rank_coarse2fine(self, query_idx, query_feats_list, gallery_feats_list, thresholds):
        '''
        '''

        for ii, (query_feats, gallery_feats) in enumerate(zip(query_feats_list, gallery_feats_list)):
            if ii == 0:  # coarse search with shortest code, e.g. 32
                code_len = len(query_feats[0])*4
                threshold = thresholds[code_len]
                topk_index, coarse_index = self.hammingsimilarity_countingsort(query_feats[query_idx], gallery_feats, code_len, threshold)

            else:  # refine with longer codes, e.g. 128, 512, 2048
                code_len = len(query_feats[0])*4
                threshold = thresholds[code_len]
                # select feature
                gallery_feats = [gallery_feats[idx] for idx in topk_index]
                # compute hamming distance and sort by counting-sort algorithm
                tmp1, tmp2 = self.hammingsimilarity_countingsort(query_feats[query_idx], gallery_feats, code_len, threshold)
                # update rank list
                refined_index = [topk_index[idx] for idx in tmp2]
                coarse_index[:len(refined_index)] = refined_index
                topk_index = [topk_index[idx] for idx in tmp1]
                # print('coarse_index:', len(coarse_index), len(np.unique(np.array(coarse_index))))
                # print('topk_index:', len(topk_index), len(np.unique(np.array(topk_index))))

        final_rank_list = coarse_index
        return final_rank_list


    def evaluate(self, query_idx, query_cam, query_label, gallery_cam, gallery_label, refined_index):
        #
        junk_index_1 = self.in1d(np.argwhere(query_label[query_idx] == gallery_label), np.argwhere(query_cam[query_idx] == gallery_cam))
        junk_index_2 = np.argwhere(gallery_label == -1)
        junk_index = np.append(junk_index_1, junk_index_2)
        #
        good_index = self.in1d(np.argwhere(query_label[query_idx] == gallery_label), np.argwhere(query_cam[query_idx] != gallery_cam))
        index_wo_junk = self.notin1d(refined_index, junk_index)
        #
        return self.compute_AP(index_wo_junk, good_index)


    def compute_AP(self, index, good_index):
        '''
        :param index: np.array, 1d
        :param good_index: np.array, 1d
        :return:
        '''

        num_good = len(good_index)
        hit = np.in1d(index, good_index)
        index_hit = np.argwhere(hit == True).flatten()

        if len(index_hit) == 0:
            AP = 0
            cmc = np.zeros([len(index)])
        else:
            precision = []
            for i in range(num_good):
                precision.append(float(i+1) / float((index_hit[i]+1)))
            AP = np.mean(np.array(precision))
            cmc = np.zeros([len(index)])
            cmc[index_hit[0]: ] = 1

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

    def hamming_dist(self, x, y):
        '''
        :param x: hex_str, e.g. 'f012d208' (a 32bits binary code)
        :param y: list, whose elements are hex_str
        :return: list
        '''
        results = [None] * len(y)
        for idx, yi in enumerate(y):
            results[idx] = hamming_distance(x, yi)
        return results

    def hammingsimilarity_countingsort(self, x, y, max_dist, threshold):
        '''
        jointly compute hamming distance and counting sort
        return the sorting results, and the top result whose distances are less than the threshold
        :param x: hex_str, e.g. 'f012d208' (a 32bits binary code)
        :param y: list, whose elements are hex_str
        :param max_dist: int, equal to code length
        :param threshold: int, the samples with distances less than it should be further refined
        '''
        # compute hamming distance and counting sort
        results = [[] for _ in range(max_dist + 1)]
        for idx, yi in enumerate(y):
            results[hamming_distance(x, yi)].append(idx)
        # return final sorting result and topk sorting result
        final_result = []
        for idx, value in enumerate(results):
            final_result += value
            if idx == threshold:
                top_result = copy.deepcopy(final_result)
        return top_result, final_result



class ThresholdOptimization:
    """
    Args:
        query_feats_list(list of np.ndarray): element dimension is [num_query, code_len]
        galler_feats_list(list of np.ndarray): element dimension is [num_gallery, code_len]
        query_pids(np.array): [num_query]
        query_pids(np.array): [num_gallery]
    Return:
        {codelen_1: threshold_1, codelen_2: threshold_2}
    """

    def __init__(self, beta):

        self.beta2 = beta**2
        self.reset()

    def reset(self):
        self.a1 = None
        self.b1 = None
        self.c1 = None
        self.a2 = None
        self.b2 = None
        self.c2 = None

    def optimize(self, query_feats_list, gallery_feats_list, query_pids, gallery_pids):
        thresholds = {}
        for query_feats, gallery_feats in zip(query_feats_list, gallery_feats_list):
            # code length
            assert query_feats.shape[1] == gallery_feats.shape[1], \
                'expect x and y have the same dimmension, but got x {} and y {}'.format(query_feats.shape[1], gallery_feats.shape[1])
            code_len = query_feats.shape[1]
            # compute hamming distances
            dists = lightreid.utils.hamming_distance(query_feats, gallery_feats)
            # record positive pairs distances
            hit_dists = []
            for q_idx, q_pid in enumerate(query_pids):
                hit_idx = np.where(q_pid == gallery_pids)
                hit_dists.append(dists[q_idx][hit_idx])
            hit_dists = np.concatenate(hit_dists)
            # compute distance thresholds of different code
            threshold = self.threshold_optimization(hit_dists, dists, code_len)
            thresholds[code_len] = threshold
            self.reset()
        return thresholds

    def threshold_optimization(self, hit_dists, dists, length):
        # global beta2
        # self.beta2 = beta ** 2

        y, x = np.histogram(hit_dists.reshape([-1]), bins=np.arange(0, length + 2))
        y = y / y.sum()  # normalize distribution
        popt, _ = curve_fit(self.gaussian, x[:length + 1], y, p0=[length / 2, 1, max(y)])
        # global a1, b1, c1
        self.a1, self.b1, self.c1 = popt

        y, x = np.histogram(dists.reshape([-1]), bins=np.arange(0, length + 2))
        y = y / y.sum()  # normalize distribution
        popt, _ = curve_fit(self.gaussian, x[:length + 1], y, p0=[length / 2, 1, max(y)])
        # global a2, b2, c2
        self.a2, self.b2, self.c2 = popt

        res = minimize_scalar(self.nfbscore, bounds=(0, length), method='bounded')
        threshold = int(res.x)
        return threshold

    def nfbscore(self, x):
        return -(self.c1 * (self.beta2 + 1) * (special.erf(scipy.sqrt(2) * (self.a1 - x) / (2 * self.b1)) - 1) /
                 (-2 * self.beta2 + self.c1 * (special.erf(scipy.sqrt(2) * (self.a1 - x) / (2 * self.b1)) - 1) +
                  self.c2 * (special.erf(scipy.sqrt(2) * (self.a2 - x) / (2 * self.b2)) - 1)))

    def gaussian(self, x, a, b, c):
        return c / (b * np.sqrt(2 * np.pi)) * np.exp(-(x - a) ** 2 / (2 * b ** 2))