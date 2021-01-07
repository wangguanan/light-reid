"""
@author:    wangguanan
@contact:   guan.wang0706@gmail.com
"""

import os, copy
from .reid_samples import ReIDSamples
import torchvision


class PartialILIDS(ReIDSamples):
    """Partial Ilids
    Only include query and gallery dataset
    Suppose all query images belong to camera0, and gallery images camera1
    """

    def __init__(self, data_path, combineall=False, download=False, **kwargs):
        assert combineall is False, \
            'unsupport combineall for {} dataset'.format(self.__class__.__name__)
        assert download is False, \
            'unsupport download, please automatically download {} dataset'.format(self.__class__.__name__)

        self.data_path = data_path
        query = self._get_probe_samples()
        gallery = self._get_gallery_samples()
        train = None
        super(PartialILIDS, self).__init__(train, query, gallery)

    def _get_probe_samples(self):
        samples = []
        f = open(os.path.join(self.data_path, 'Probe.txt'))
        for line in f.readlines():
            line = line.replace('\n', '')
            image_path = line
            pid = int(line.split('/')[1].replace('.jpg', ''))
            samples.append([os.path.join(self.data_path, image_path), pid, 0])
        return samples

    def _get_gallery_samples(self):
        samples = []
        f = open(os.path.join(self.data_path, 'Gallery.txt'))
        for line in f.readlines():
            line = line.replace('\n', '')
            image_path = line
            pid = int(line.split('/')[1].replace('.jpg', ''))
            samples.append([os.path.join(self.data_path, image_path), pid, 1])
        return samples

