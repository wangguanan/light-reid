"""
@author:    wangguanan
@contact:   guan.wang0706@gmail.com
"""

import numpy as np
import copy
from PIL import Image
import torch.utils.data as data

from .samplers import PKSampler


class ReIDDataset:

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        sample = copy.deepcopy(self.samples[index])
        sample[0] = self._loader(sample[0])
        if self.transform is not None:
            sample[0] = self.transform(sample[0])
        sample[1] = np.array(sample[1])
        return sample

    def __len__(self):
        return len(self.samples)

    def _loader(self, img_path):
        return Image.open(img_path).convert('RGB')


class DataManager(object):
    '''
    Args:
        sources(list): tuples of torch.data.ReIDDataset, source datasets to train with
        target(torch.data.ReIDDataset): target dataset to evaluate on
        transforms_train(torch.torchvision.transforms):
        transforms_test(torch.torchvision.transforms):
        sampler(str): sample strategy for train dataset, support 'pk' and 'random'.
            when 'pk', params 'p' and 'k' must be given.
            when 'random', params 'batch_size' must be given.
    Example:
        datamanager = DataManager(
            sources=[lightreid.data.Market1501(data_path='', combineall=False), lightreid.data.DukeMTMCreID(data_path='', combineall=False)],
            target=lightreid.data.Market1501(data_path='', combineall=False),
            transforms_train=lightreid.data.build_transforms(img_size=[256,128], transform_list=['randomflip', 'padcrop', 'colorjitor', 'rea']),
            transforms_test=lightreid.data.build_transforms(img_size=[256,128], transform_list=[]),
            sampler='pk', p=16, k=4
            )
        train_loader = datamanager.train_loader
        query_loader = datamanager.query_loader
        gallery_loader = datamanager.gallery_loader
    '''

    KWARGS = ['batch_size', 'p', 'k']
    SAMPLERS = ['random', 'pk']

    def __init__(self, sources, target, transforms_train, transforms_test, sampler, **kwargs):

        # check param sample and kwargs is legal
        assert sampler in DataManager.SAMPLERS, \
            'sampler expect {}. but got {}'.format(DataManager.SAMPLERS, sampler)

        # init train/query/gallery dataset
        train = self.combine([source.train for source in sources])
        self.class_num = len(set([sample[1] for sample in train]))
        self.train_dataset = ReIDDataset(train, transforms_train)

        self.query_gallery_dataset_dict = {}
        for val in target:
            query_dataset = ReIDDataset(val.query, transforms_test)
            gallery_dataset = ReIDDataset(val.gallery, transforms_test)
            self.query_gallery_dataset_dict[val.__class__.__name__] = (query_dataset, gallery_dataset)

        # train loader
        if sampler == 'random':
            assert 'batch_size' in kwargs.keys(), 'param batch_size(int) must be given when sample=\'random\''
            batch_size = kwargs['batch_size']
            self.train_loader = data.DataLoader(self.train_dataset, batch_size=batch_size, num_workers=8, drop_last=True, shuffle=True)
        elif sampler == 'pk':
            assert 'p' in kwargs.keys() and 'k' in kwargs.keys(), 'param p(int) and k(int) must be given when sample=\'random\''
            p, k = kwargs['p'], kwargs['k']
            self.train_loader = data.DataLoader(self.train_dataset, batch_size=p*k, num_workers=8, drop_last=True, sampler=PKSampler(self.train_dataset, k=k))
        else:
            assert 0, 'expect {}. but got {}'.format(DataManager.SAMPLERS, sampler)

        # query and gallery loader
        self.query_gallery_loader_dict = {}
        for dataset_name, (query_dataset, gallery_dataset) in self.query_gallery_dataset_dict.items():
            query_loader = data.DataLoader(query_dataset, batch_size=64, num_workers=8, drop_last=False, shuffle=False)
            gallery_loader = data.DataLoader(gallery_dataset, batch_size=64, num_workers=8, drop_last=False, shuffle=False)
            self.query_gallery_loader_dict[dataset_name] = (query_loader, gallery_loader)


    def combine(self, samples_list):
        '''combine more than one samples (e.g. market.train and duke.train) as a samples'''
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
