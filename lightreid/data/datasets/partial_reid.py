"""
@author:    wangguanan
@contact:   guan.wang0706@gmail.com
"""

import os,copy
import os.path as osp

from .reid_samples import ReIDSamples
from lightreid.utils import os_walk


class PartialReID(ReIDSamples):
    """PartialReID.
    Reference:
    """

    def __init__(self, data_path, combineall=False, download=False, **kwargs):
        assert combineall is False, \
            'unsupport combineall for {} dataset'.format(self.__class__.__name__)
        assert download is False, \
            'unsupport download, please automatically download {} dataset'.format(self.__class__.__name__)
        query = self._get_probe_samples(data_path)
        gallery = self._get_gallery_samples(data_path)
        train = None
        super(PartialReID, self).__init__(train, query, gallery)        

    def _get_probe_samples(self, data_path):
        samples = []
        f = open(os.path.join(data_path, 'Probe.txt'))
        for line in f.readlines():
            line = line.replace('\n', '')
            image_path = line
            pid = int(line.split('/')[1].replace('.jpg', '').split('_')[0])
            samples.append([os.path.join(data_path, image_path), pid, 0])
        return samples

    def _get_gallery_samples(self, data_path):
        """
        Return:
            samples(list): [
                [img1_path, id1, 1], # suppose all gallery images belongs to camera 1
                [img2_path, id2, 1],
                ...
            ]
        """
        samples = []
        f = open(os.path.join(data_path, 'Gallery.txt'))
        for line in f.readlines():
            line = line.replace('\n', '')
            image_path = line
            pid = int(line.split('/')[1].replace('.jpg', '').split('_')[0])
            samples.append([os.path.join(data_path, image_path), pid, 1])
        return samples
