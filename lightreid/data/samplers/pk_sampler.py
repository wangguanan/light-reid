"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import itertools

import torch.utils.data as data
import random
import torch.distributed as dist


class PKSampler(data.sampler.Sampler):
    '''
    PK sample according to person identity
    Arguments:
        data_source(lightreid.data.ReIDdataset)
        k(int): sample k images of each person
    '''

    def __init__(self, data_source, k):
        # parameters
        self.data_source = data_source
        self.pid_idx = 1
        self.k = k
        # multi-processing
        # self.mp = dist.is_available()
        self.mp = False
        if self.mp:
            self.rank = dist.get_rank()
            self.word_size = dist.get_world_size()
        # init
        self.samples = self.data_source.samples
        self.class_dict = self._tuple2dict(self.samples)

    def __iter__(self):
        self.sample_list = self._generate_list(self.class_dict)
        if not self.mp:
            return iter(self.sample_list)
        else:
            start = self.rank
            return itertools.islice(self.sample_list, start, None, self.word_size)

    def __len__(self):
        return len(self.sample_list)

    def _tuple2dict(self, inputs):
        '''
        :param inputs: list with tuple elemnts, [(image_path1, class_index_1), (imagespath_2, class_index_2), ...]
        :return: dict, {class_index_i: [samples_index1, samples_index2, ...]}
        '''
        dict = {}
        for index, each_input in enumerate(inputs):
            class_index = each_input[self.pid_idx]
            if class_index not in list(dict.keys()):
                dict[class_index] = [index]
            else:
                dict[class_index].append(index)
        return dict

    def _generate_list(self, dict):
        sample_list = []

        dict_copy = dict.copy()
        keys = list(dict_copy.keys())
        random.shuffle(keys)
        while len(sample_list) < len(self.samples):
            for key in keys:
                if len(sample_list) >= len(self.samples): break
                value = dict_copy[key]
                if len(value) >= self.k:
                    random.shuffle(value)
                    sample_list.extend(value[0: self.k])
                else:
                    value = value * self.k
                    random.shuffle(value)
                    sample_list.extend(value[0: self.k])
        return sample_list
