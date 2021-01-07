import numpy as np
import copy
from hexhamming import hamming_distance
import progressbar
from sklearn import metrics as sk_metrics
import time

from lightreid.utils.meters import AverageMeter
from ..build import EVALUATORs_REGISTRY


__all__ = ['CmcMapEvaluator1b1']


class FileTransferFrequency(progressbar.FileTransferSpeed):

    def _speed(self, value, elapsed):
        speed = elapsed / float(value)
        return speed * 1000, 1


def func(x, sub_y, start_idx, suby_len):
    result = [None for _ in range(suby_len)]
    for idx, yi in enumerate(sub_y):
        result[idx] = (hamming_distance(x, yi), start_idx+idx)
    return result


@EVALUATORs_REGISTRY.register()
class CmcMapEvaluator1b1:
    '''
    Evaluate every query one-by-one.
    This is more practical in real-world Re-ID application.
    Compute Rank@k and mean Average Precision (mAP) scores
    '''

    def __init__(self, metric, mode):

        assert metric in ['cosine', 'euclidean', 'hamming']
        assert mode in ['inter-camera', 'intra-camera', 'all']
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



    def compute(self, query_feats, query_camids, query_pids, gallery_feats, gallery_camids, gallery_pids, return_time=False):
        '''rank and evaluate'''

        if self.metric == 'hamming': # convert np.ndarray to hex
            assert type(query_feats) == type(gallery_feats)
            if isinstance(query_feats, np.ndarray):
                query_hex, gallery_hex = [], []
                code_len = query_feats.shape[1]
                for query_feature in query_feats:
                    binary_str = ''.join(str(int(i)) for i in query_feature)
                    hex_str = hex(int(binary_str, 2))[2:].zfill(int(code_len / 4))
                    query_hex.append(hex_str)
                for gallery_feat in gallery_feats:
                    binary_str = ''.join(str(int(i)) for i in gallery_feat)
                    hex_str = hex(int(binary_str, 2))[2:].zfill(int(code_len / 4))
                    gallery_hex.append(hex_str)
                query_feats = query_hex
                gallery_feats = gallery_hex

        # rank
        ranktime_meter = AverageMeter()
        all_rank_list = np.zeros([len(query_pids), len(gallery_pids)])
        for query_idx in self.bar_rank(range(len(query_pids))):
            ts = time.time()
            rank_list = self.rank(query_idx, query_feats, gallery_feats)
            ranktime_meter.update(time.time() - ts)
            all_rank_list[query_idx, :] = rank_list

        # # rank with multiprocessing
        # pool = multiprocessing.Pool()
        # # cpus = multiprocessing.cpu_count()
        # cpus = 4
        # gallery_len = len(gallery_feats)
        # split_len = int(gallery_len / cpus)
        #
        # split_startend_idx = []
        # start_idx = 0
        # while True:
        #     end_idx = start_idx + split_len
        #     if end_idx >= gallery_len:
        #         end_idx = gallery_len
        #         split_startend_idx.append((start_idx, end_idx, end_idx-start_idx))
        #         break
        #     else:
        #         split_startend_idx.append((start_idx, end_idx, split_len))
        #     start_idx = end_idx
        #
        # all_rank_list = np.zeros([len(query_pids), len(gallery_pids)])
        # for query_idx in self.bar_rank(range(len(query_pids))):
        #     rank_list = self.rank(query_idx, query_feats, gallery_feats, split_startend_idx, pool)
        #     all_rank_list[query_idx, :] = rank_list

        # evaluate
        evaltime_meter = AverageMeter()
        APs, CMC = [], []
        for query_idx in self.bar_evaluate(range(len(query_pids))):
            ts = time.time()
            AP, cmc = self.evaluate(
                query_idx, query_camids, query_pids,
                gallery_camids, gallery_pids, np.array(all_rank_list[query_idx]))
            evaltime_meter.update(time.time() - ts)
            # record
            APs.append(AP); CMC.append(cmc)

        # compute CMC and mAP
        MAP = np.array(APs).mean()
        min_len = min([len(cmc) for cmc in CMC])
        CMC = [cmc[:min_len] for cmc in CMC]
        CMC = np.mean(np.array(CMC), axis=0)

        if return_time:
            return MAP, CMC, ranktime_meter.get_val(), evaltime_meter.get_val()
        return MAP, CMC

    def rank(self, query_idx, query_features, gallery_features, split_startend_idx=None, pool=None):
        if self.metric is 'hamming':
            code_len = 4*len(query_features[0])
            if pool is None:
                _, rank_results = self.hammingsimilarity_countingsort(
                    query_features[query_idx], gallery_features, code_len, None)
            else:
                _, rank_results = self.hammingsimilarity_countingsort_multiprocess(
                    query_features[query_idx], gallery_features, code_len, split_startend_idx, pool)
        elif self.metric is 'cosine':
            distance = sk_metrics.pairwise.cosine_distances(np.expand_dims(query_features[query_idx, :], axis=0), gallery_features).squeeze(axis=0)
            rank_results = np.argsort(distance)
        elif self.metric is 'euclidean':
            distance = sk_metrics.pairwise.euclidean_distances(np.expand_dims(query_features[query_idx, :], axis=0), gallery_features).squeeze(axis=0)
            rank_results = np.argsort(distance)
        return rank_results

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
            hamming_distance(x, yi)
            results[hamming_distance(x, yi)].append(idx)
        # return final sorting result and topk sorting result
        final_result = []
        for idx, value in enumerate(results):
            final_result += value
            top_result = None
            if idx == threshold:
                top_result = copy.deepcopy(final_result)
        return top_result, final_result


    def hammingsimilarity_countingsort_multiprocess(self, x, y, max_dist, split_startend_idx, pool):
        '''
        jointly compute hamming distance and counting sort
        return the sorting results, and the top result whose distances are less than the threshold
        :param x: hex_str, e.g. 'f012d208' (a 32bits binary code)
        :param y: list, whose elements are hex_str
        :param max_dist: int, equal to code length
        :param threshold: int, the samples with distances less than it should be further refined
        '''

        results = []
        for start_idx, end_idx, suby_len in split_startend_idx:
            sub_y = y[start_idx: end_idx]
            result = pool.apply_async(func, args=(x, sub_y, start_idx, suby_len,))
            results.append(result)

        # pool.close()
        # pool.join()

        xxx = [[] for _ in range(max_dist + 1)]
        for result in results:
            result = result.get()
            for x,y in result:
                xxx[x].append(y)

        results = []
        for xx in xxx:
            results += xx

        return None, results