

import numpy as np
from sklearn import metrics as sk_metrics

class PersonReIDMAP:
    '''
    Compute Rank@k and mean Average Precision (mAP) scores
    Used for Person ReID
    Test on MarKet and Duke
    '''

    def __init__(self, query_feature, query_cam, query_label, gallery_feature, gallery_cam, gallery_label, dist):
        '''
        :param query_feature: np.array, bs * feature_dim
        :param query_cam: np.array, 1d
        :param query_label: np.array, 1d
        :param gallery_feature: np.array, gallery_size * feature_dim
        :param gallery_cam: np.array, 1d
        :param gallery_label: np.array, 1d
        '''

        self.query_feature = query_feature
        self.query_cam = query_cam
        self.query_label = query_label
        self.gallery_feature = gallery_feature
        self.gallery_cam = gallery_cam
        self.gallery_label = gallery_label

        assert dist in ['cosine', 'euclidean']
        self.dist = dist

        # normalize feature for fast cosine computation
        if self.dist == 'cosine':
            self.query_feature = self.normalize(self.query_feature)
            self.gallery_feature = self.normalize(self.gallery_feature)

        APs = []
        CMC = []
        for i in range(len(query_label)):
            AP, cmc = self.evaluate(self.query_feature[i], self.query_cam[i], self.query_label[i],
                                    self.gallery_feature, self.gallery_cam, self.gallery_label)
            APs.append(AP)
            CMC.append(cmc)
            # print('{}/{}'.format(i, len(query_label)))

        self.APs = np.array(APs)
        self.mAP = np.mean(self.APs)

        min_len = 99999999
        for cmc in CMC:
            if len(cmc) < min_len:
                min_len = len(cmc)
        for i, cmc in enumerate(CMC):
            CMC[i] = cmc[0: min_len]
        self.CMC = np.mean(np.array(CMC), axis=0)

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

    def evaluate(self, query_feature, query_cam, query_label, gallery_feature, gallery_cam, gallery_label):
        '''
        :param query_feature: np.array, 1d
        :param query_cam: int
        :param query_label: int
        :param gallery_feature: np.array, 2d, gallerys_size * feature_dim
        :param gallery_cam: np.array, 1d
        :param gallery_label: np.array, 1d
        :return:
        '''

        # cosine score
        if self.dist is 'cosine':
            # feature has been normalize during intialization
            score = np.matmul(query_feature, gallery_feature.transpose())
            index = np.argsort(score)[::-1]
        elif self.dist is 'euclidean':
            score = self.l2(query_feature.reshape([1, -1]), gallery_feature)
            index = np.argsort(score.reshape([-1]))

        junk_index_1 = self.in1d(np.argwhere(query_label == gallery_label), np.argwhere(query_cam == gallery_cam))
        junk_index_2 = np.argwhere(gallery_label == -1)
        junk_index = np.append(junk_index_1, junk_index_2)

        good_index = self.in1d(np.argwhere(query_label == gallery_label), np.argwhere(query_cam != gallery_cam))
        index_wo_junk = self.notin1d(index, junk_index)

        return self.compute_AP(index_wo_junk, good_index)

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

    def normalize(self, x):
        norm = np.tile(np.sqrt(np.sum(np.square(x), axis=1, keepdims=True)), [1, x.shape[1]])
        return x / norm

    def cosine_dist(self, x, y):
        return sk_metrics.pairwise.cosine_distances(x, y)

    def euclidean_dist(self, x, y):
        return sk_metrics.pairwise.euclidean_distances(x, y)
