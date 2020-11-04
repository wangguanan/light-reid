"""
@author:    wangguanan
@contact:   guan.wang0706@gmail.com
"""

import os, copy
from .reid_samples import ReIDSamples


class AirportAlert(ReIDSamples):
    """AirportAlert.
    Reference:
    Dataset statistics:
        - identities: 9651
        - images: 39902.
    Args:
        data_path(str): path to AirportAlert dataset
        combineall(bool): combine train and test sets as train set if True
    """

    def __init__(self, data_path, combineall=False, download=False):

        imgpath_file = os.path.join(data_path, 'filepath.txt')
        personid_file = os.path.join(data_path, 'personID.txt')
        camid_file = os.path.join(data_path, 'camID.txt')
        with open(camid_file, 'r') as f:
            cam_ids = f.readlines()
            cam_ids = [int(v.replace('\n', '')) for v in cam_ids]
        with open(personid_file, 'r') as f:
            person_ids = f.readlines()
            person_ids = [int(v.replace('\n', '')) for v in person_ids]
        with open(imgpath_file, 'r') as f:
            img_paths = f.readlines()
            img_paths = [v.replace('\n', '').replace('\\', '/') for v in img_paths]

        train = []
        for imgpath, pid, camid in zip(img_paths, person_ids, cam_ids):
            if pid == 0 or pid >=1320011: continue
            train.append([os.path.join(data_path, imgpath), pid, camid])
        query = None
        gallery = None

        # init
        super(AirportAlert, self).__init__(train, query, gallery)


if __name__ == '__main__':
    AirportAlert('/data/datasets/airport')