"""
@author:    wangguanan
@contact:   guan.wang0706@gmail.com
"""

import os
import copy
import numpy as np
from PIL import Image
from prettytable import PrettyTable


class ReIDSamples:
    '''
    An abstract class representing a Re-ID samples.
    Return:
        train (list): contains tuples of (img_path(s), pid, camid).
        query (list): contains tuples of (img_path(s), pid, camid).
        gallery (list): contains tuples of (img_path(s), pid, camid).
    '''

    def __init__(self):
        self.train = None
        self.query = None
        self.gallery = None

    def statistics(self, train, query, gallery, name=None):
        '''show samples statistics'''

        def analyze(samples):
            pid_num = len(set([sample[1] for sample in samples]))
            cid_num = len(set([sample[2] for sample in samples]))
            sample_num = len(samples)
            return sample_num, pid_num, cid_num

        train_info = analyze(train)
        query_info = analyze(query)
        gallery_info = analyze(gallery)

        # please kindly install prettytable: ```pip install prettyrable```
        table = PrettyTable(['set', 'images', 'identities', 'cameras'])
        table.add_row([self.__class__.__name__ if name is None else name, '', '', ''])
        table.add_row(['train', str(train_info[0]), str(train_info[1]), str(train_info[2])])
        table.add_row(['query', str(query_info[0]), str(query_info[1]), str(query_info[2])])
        table.add_row(['gallery', str(gallery_info[0]), str(gallery_info[1]), str(gallery_info[2])])
        print(table)

    def os_walk(self, folder_dir):
        for root, dirs, files in os.walk(folder_dir):
            files = sorted(files, reverse=True)
            dirs = sorted(dirs, reverse=True)
            return root, dirs, files

    def relabel(self, samples):
        '''relabel person identities'''
        ids = list(set([sample[1] for sample in samples]))
        ids.sort()
        for sample in samples:
            sample[1] = ids.index(sample[1])
        return samples
