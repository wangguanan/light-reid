"""
@author:    wangguanan
@contact:   guan.wang0706@gmail.com
"""

import os, copy
import numpy as np
from .reid_samples import ReIDSamples


class RAP(ReIDSamples):

    dataset_url = None

    def __init__(self, data_path='', **kwargs):
        '''
        :param data_path:
        :return: [(path, identiti_id, camera_id)]
        '''
        import scipy.io as scio
        mat_path = os.path.join(data_path, 'train/RAP_reid_data.mat')
        imgs_path = os.path.join(data_path, 'train/training_images/')
        mat = scio.loadmat(mat_path)['RAP_reid_data']
        train_samples = []
        for idx in range(len(mat[0, 0][0])):
            img = mat[0, 0][0][idx][0][0]
            pid = mat[0, 0][0][idx][1][0][0]
            cid = mat[0, 0][0][idx][2][0]
            train_samples.append([os.path.join(imgs_path, img), int(pid), int(cid.replace('CAM', ''))])

        # init
        train = copy.deepcopy(train_samples)
        query = None
        gallery = None
        super(RAP, self).__init__(train, query, gallery)

    def split(self, samples, ratio, seed=None):
        if seed is not None:
            np.random.seed(seed)
        ids = [sample[1] for sample in samples]
        ids = list(set(ids))
        selected_ids = np.random.choice(ids, int(len(ids)*ratio), replace=False)
        samples1, samples2 = [], []
        for sample in samples:
            if sample[1] in selected_ids:
                samples1.append(sample)
            else:
                samples2.append(sample)
        return samples1, samples2
