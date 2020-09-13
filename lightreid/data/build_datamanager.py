"""
author: Guan'an Wang
e-mail: guan.wang0706@gmail.com
"""

from .datasets import build_train_dataset, build_test_dataset
from .transforms import build_transforms
from .data_manager import DataManager

def build_datamanager(
        sources, targets,
        img_size, transforms_train, transforms_test,
        sampler,
        **kwargs):
    """
    Args:
        sources(list): list of dataset names
        targets(list): list of dataset names
        img_size(tuple): [height, width]
        transforms_train(list): list of tranforms(str)
    """
    combineall = kwargs['combineall'] if 'combineall' in kwargs.keys() else False
    train_dataset = build_train_dataset(sources, combineall)
    test_dataset = build_test_dataset(targets)
    transforms_train = build_transforms(img_size=img_size, transforms_list=transforms_train)
    transforms_test = build_transforms(img_size=img_size, transforms_list=transforms_test)
    return DataManager(train_dataset, test_dataset, transforms_train, transforms_test, sampler, **kwargs)
