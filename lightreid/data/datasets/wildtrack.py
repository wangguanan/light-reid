"""
@author:    wangguanan
@contact:   guan.wang0706@gmail.com
"""

import os, copy
import os.path as osp

from .reid_samples import ReIDSamples
from lightreid.utils import os_walk

class WildTrackCrop(ReIDSamples):
    """WildTrack.
    Reference:
        WILDTRACK: A Multi-camera HD Dataset for Dense Unscripted Pedestrian Detection
            T. Chavdarova; P. Baqué; A. Maksai; S. Bouquet; C. Jose et al.
    URL: `<https://www.epfl.ch/labs/cvlab/data/data-wildtrack/>`_
    Dataset statistics:
        - identities: 313
        - images: 33979 (train only)
        - cameras: 7
    Args:
        data_path(str): path to WildTrackCrop dataset
        combineall(bool): combine train and test sets as train set if True
    """
    dataset_url = None

    def __init__(self, data_path, download=False, **kwargs):

        # is not exist and download true, download dataset or stop
        if not os.path.exists(data_path):
            if download:
                print('dataset path {} is not existed, start download dataset'.format(data_path))
                self.download_dataset(data_path, self.dataset_url)
            else:
                return 'dataset path {} is not existed'

        # load sample
        root1, folders, _ = os_walk(data_path)
        samples = []
        for folder in folders:
            pid = int(folder)
            root2, _, files = os_walk(osp.join(root1, folder))
            files = [file for file in files if '.png' in file or '.jpg' in file]
            for file in files:
                cam_id = int(file.split('_')[0])
                img_path = osp.join(root2, file)
                samples.append([img_path, pid, cam_id])

        # init
        train = samples
        query = None
        gallery = None
        super(WildTrackCrop, self).__init__(train, query, gallery)

    def _analysis_file_name(self, file_name):
        '''
        :param file_name: format like 02_00000135.png
        :return: 2
        '''
        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        person_id, camera_id = int(split_list[0]), int(split_list[1])
        return person_id, camera_id