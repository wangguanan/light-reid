"""
@author:    wangguanan
@contact:   guan.wang0706@gmail.com
"""

import os, copy
from .reid_samples import ReIDSamples


class MSMT17(ReIDSamples):
    """MSMT17.
    Reference:
        Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.
    URL: `<http://www.pkuvmc.com/publications/msmt17.html>`_
    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    Args:
        data_path(str): path to MSMT17 dataset
        combineall(bool): combine train and test sets as train set if True
    """

    dataset_url = None

    def __init__(self, data_path, combineall=False, download=False):

        # is not exist and download true, download dataset or stop
        if not os.path.exists(data_path):
            if download:
                print('dataset path {} is not existed, start download dataset'.format(data_path))
                self.download_dataset(data_path, self.dataset_url)
            else:
                return 'dataset path {} is not existed, start download dataset'.format(data_path)

        list_train_path = os.path.join(data_path, 'list_train.txt')
        list_val_path = os.path.join(data_path, 'list_val.txt')
        list_query_path = os.path.join(data_path, 'list_query.txt')
        list_gallery_path = os.path.join(data_path, 'list_gallery.txt')

        train = self._load_list(os.path.join(data_path, 'train/'), list_train_path)
        val = self._load_list(os.path.join(data_path, 'train/'), list_val_path)
        query = self._load_list(os.path.join(data_path, 'test/'), list_query_path)
        gallery = self._load_list(os.path.join(data_path, 'test/'), list_gallery_path)
        train = copy.deepcopy(train) + copy.deepcopy(val)

        # init
        super(MSMT17, self).__init__(train, query, gallery, combineall)

    def _load_list(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        data = []
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2]) - 1  # index starts from 0
            img_path = os.path.join(dir_path, img_path)
            data.append([img_path, pid, camid])
        return data

    def _combine_samples(self, samples_list):
        '''combine more than one samples (e.g. msmt.train and msmt.query) as a samples'''
        all_samples = []
        max_pid, max_cid = 0, 0
        for samples in samples_list:
            for a_sample in samples:
                img_path = a_sample[0]
                pid = max_pid + a_sample[1]
                cid = max_cid + a_sample[2]
                all_samples.append([img_path, pid, cid])
            max_pid = max([sample[1] for sample in all_samples])
            max_cid = max([sample[2] for sample in all_samples])
        return all_samples
