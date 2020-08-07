import numpy as np
import copy
from hexhamming import hamming_distance
import progressbar


__all__ = ['CmcMapEvaluator1b1']


class FileTransferFrequency(progressbar.FileTransferSpeed):

    def _speed(self, value, elapsed):
        speed = elapsed / float(value)
        return speed * 1000, 1


class CmcMapEvaluator1b1:
    '''
    Evaluate every query one-by-one.
    This is more practical in real-world Re-ID application.
    Compute Rank@k and mean Average Precision (mAP) scores
    '''

    def __init__(self, metric, mode, ):

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


    def compute(self, query_feats, query_camids, query_pids, gallery_feats, gallery_camids, gallery_pids, thresholds=None):
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
            elif isinstance(query_feats, list):
                query_hex_list = []
                gallery_hex_list = []
                for a_query_feats, a_gallery_feats in zip(query_feats, gallery_feats):
                    query_hex, gallery_hex = [], []
                    for query_feature in a_query_feats:
                        binary_str = ''.join(str(int(i)) for i in query_feature)
                        hex_str = hex(int(binary_str, 2))[2:].zfill(int(code_len / 4))
                        query_hex.append(hex_str)
                    for gallery_feat in a_gallery_feats:
                        binary_str = ''.join(str(int(i)) for i in gallery_feat)
                        hex_str = hex(int(binary_str, 2))[2:].zfill(int(code_len / 4))
                        gallery_hex.append(hex_str)
                    query_hex_list.append(query_hex)
                    gallery_hex_list.append(gallery_hex)
                query_feats = query_hex_list
                gallery_feats = gallery_hex_list

        # rank
        all_rank_list = []
        for query_idx in self.bar_rank(range(len(query_pids))):
            if isinstance(query_feats, list) or isinstance(query_feats, np.ndarray): # vanilla eval
                rank_list = self.rank(query_idx, query_feats, gallery_feats)
            elif isinstance(query_feats, list): # eval with coarse2fine
                rank_list = self.rank_coarse2fine(query_idx, query_feats, gallery_feats)
            else:
                assert 0, 'expect query/gallery features type np.ndarray and dict, got {} and {}'.\
                    format(type(query_feats), type(gallery_feats))
            all_rank_list.append(rank_list)

        # evaluate
        APs, CMC = [], []
        for query_idx in self.bar_evaluate(range(len(query_pids))):
            AP, cmc = self.evaluate(
                query_idx, query_camids, query_pids,
                gallery_camids, gallery_pids, np.array(all_rank_list[query_idx]))
            # record
            APs.append(AP); CMC.append(cmc)

        # compute CMC and mAP
        MAP = np.array(APs).mean()
        min_len = min([len(cmc) for cmc in CMC])
        CMC = [cmc[:min_len] for cmc in CMC]
        CMC = np.mean(np.array(CMC), axis=0)

        return MAP, CMC


    def rank(self, query_idx, query_features, gallery_features):
        code_len = 4*len(query_features[0])
        _, coarse_index = self.hammingsimilarity_countingsort(
            query_features[query_idx], gallery_features, code_len, threshold=None)
        return coarse_index


    def rank_coarse2fine(self, query_idx, query_feats_list, gallery_feats_list, thresholds):
        '''
        Args:
            query_idx(int):
            query_feature_dict/gallery_feature_dict(dict):
            {'binary-code-32': np.ndarray of size [n_samples, 32],
             'binary-code-128': np.ndarray of size [n_samples, 128],
             'binary-code-512': np.ndarray of size [n_samples, 512], ...}
        '''

        for ii, (query_feats, gallery_feats) in zip(query_feats_list, gallery_feats_list):
            if ii == 0:  # coarse search with shortest code, e.g. 32
                code_len = query_feats.shape[1]
                threshold = thresholds[code_len]
                topk_index, coarse_index = self.hammingsimilarity_countingsort(query_feats[query_idx], gallery_feats, code_len, threshold)

            else:  # refine with longer codes, e.g. 128, 512, 2048
                code_len = query_feats.shape[1]
                threshold = thresholds[code_len]
                # select feature
                gallery_feat = [gallery_feat[idx] for idx in topk_index]
                # compute hamming distance and sort by counting-sort algorithm
                tmp1, tmp2 = self.hammingsimilarity_countingsort(query_feats[query_idx], gallery_feat, code_len, threshold)
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
            top_result = None
            if idx == threshold:
                top_result = copy.deepcopy(final_result)
        return top_result, final_result

