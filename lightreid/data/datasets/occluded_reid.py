"""
@author:    wangguanan
@contact:   guan.wang0706@gmail.com
"""

import os, copy
from .reid_samples import ReIDSamples
import torchvision


class OccludedReID(ReIDSamples):
    """Occluded ReID
    Only include query and gallery dataset
    Suppose all query images belong to camera0, and gallery images camera 1
    """

    def __init__(self, data_path, combineall=False, download=False, **kwargs):
        assert combineall is False, \
            'unsupport combineall for {} dataset'.format(self.__class__.__name__)
        assert download is False, \
            'unsupport download, please automatically download {} dataset'.format(self.__class__.__name__)

        self.probe_path = os.path.join(data_path, 'occluded_body_images/')
        self.gallery_path = os.path.join(data_path, 'whole_body_images/')

        probe_samples = torchvision.datasets.ImageFolder(self.probe_path).samples
        query = [list(probe_sample)+[0] for probe_sample in probe_samples]
        gallery_samples = torchvision.datasets.ImageFolder(self.gallery_path).samples
        gallery = [list(gallery_sample)+[1] for gallery_sample in gallery_samples]

        # init
        train = None
        super(OccludedReID, self).__init__(train, query, gallery)


