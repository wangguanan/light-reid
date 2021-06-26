"""
author: Guan'an Wang
e-mail: guan.wang0706@gmail.com
"""

from .datasets import build_train_dataset, build_test_dataset
from .transforms import build_transforms
from .datamanager import DataManager

def build_datamanager(
        sources, targets,
        img_size, transforms_train, transforms_test,
        sampler,
        **kwargs):
    """
    Args:
        sources(list): list of dataset names
        targets(list): list of dataset names
        img_size(tuple): (height, width)
        transforms_train(list): list of tranforms(str)
    Example:
        datamanager = lightreid.data.build_datamanager(
            sources=['market1501', 'dukemtmcreid'], targets=['market1501', 'dukemtmcreid'], img_size=(256, 128),
            transforms_train=['randomflip', 'padcrop', 'rea'], transforms_test=[],
            sampler='pk', p=16, k=4)
        for (img, pid, camid) in datamanager.train_loader:
            # train code
        for dataset_name, (query_loader, gallery_loader) in datamanager.query_gallery_loader_dict.items():
            # test code
    """
    combineall = kwargs['combineall'] if 'combineall' in kwargs.keys() else False
    train_dataset = build_train_dataset(sources, combineall)
    test_dataset = build_test_dataset(targets)
    transforms_train = build_transforms(img_size=img_size, transforms_list=transforms_train, **kwargs)
    transforms_test = build_transforms(img_size=img_size, transforms_list=transforms_test, **kwargs)
    return DataManager(train_dataset, test_dataset, transforms_train, transforms_test, sampler, **kwargs)
