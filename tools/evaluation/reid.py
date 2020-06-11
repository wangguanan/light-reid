import numpy as np
from sklearn import metrics as sk_metrics

class ReIDEvaluator:
    '''
    Compute Rank@k and mean Average Precision (mAP) scores for ReID task
    '''

    def __init__(self, dist, mode):
        assert dist in ['cosine', 'euclidean']
        self.dist = dist
        assert mode in ['inter-camera', 'intra-camera', 'all']
        self.mode = mode

    def evaluate(self, query_features, query_camids, query_pids, gallery_features, gallery_camids, gallery_pids):

        '''compute distance matrix'''
        if self.dist is 'cosine':
            scores = self.cosine_dist(query_features, gallery_features)
            rank_results = np.argsort(scores)[:, ::-1]
        elif self.dist is 'euclidean':
            scores = self.euclidean_dist(query_features, gallery_features)
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
            # self_junk = a_rank[0] if a_rank[0] == 1 or a_rank[0] == 0 else []# remove self
            self_junk = a_rank[0]
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

    def cosine_dist(self, x, y):
        '''compute cosine distance between two martrix x and y with sizes (n1, d), (n2, d)'''
        def normalize(x):
            '''normalize a 2d matrix along axis 1'''
            norm = np.tile(np.sqrt(np.sum(np.square(x), axis=1, keepdims=True)), [1, x.shape[1]])
            return x / norm
        x = normalize(x)
        y = normalize(y)
        return np.matmul(x, y.transpose([1,0]))

    # def cosine_dist(self, x, y):
    #     return sk_metrics.pairwise.cosine_distances(x, y)

    def euclidean_dist(self, x, y):
        '''compute eculidean distance between two martrix x and y with sizes (n1, d), (n2, d)'''
        return sk_metrics.pairwise.euclidean_distances(x, y)



class PrecisionRecall(ReIDEvaluator):

    def __init__(self, dist, mode):
        assert dist in ['cosine', 'euclidean']
        self.dist = dist
        assert mode in ['intra-camera', 'inter-camera', 'all']
        self.mode = mode

    def evaluate(self, thresholds, query_features, query_camids, query_pids, gallery_features, gallery_camids, gallery_pids):

        '''compute distance matrix'''
        if self.dist is 'cosine':
            scores = self.cosine_dist(query_features, gallery_features)
        elif self.dist is 'euclidean':
            scores = self.euclidean_dist(query_features, gallery_features)

        pid_similarity = (np.expand_dims(query_pids, axis=0).transpose([1,0]) == np.expand_dims(gallery_pids, axis=0)).astype(np.float)
        cid_similarity = (np.expand_dims(query_camids, axis=0).transpose([1,0]) == np.expand_dims(gallery_camids, axis=0)).astype(np.float)

        pres, recalls = [], []
        for threshold in thresholds:
            if self.dist == 'cosine':
                hits = scores >= threshold
            elif self.dist == 'euclidean':
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
