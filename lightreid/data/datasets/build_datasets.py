from .market1501 import Market1501
from .dukemtmcreid import DukeMTMCreID
from .msmt17 import MSMT17

import yaml
from os.path import realpath, dirname, join


__datasets_factory = {
    'market1501': Market1501,
    'dukemtmcreid': DukeMTMCreID,
    'msmt17': MSMT17
}


with open(join(dirname(__file__), 'config_datasetpath.yml')) as file:
    __datasets_path = yaml.load(file, Loader=yaml.FullLoader)


def build_train_dataset(dataset_list, combineall=False):
    """
    Example:
        datamanager = lightreid.data.DataManager(
            sources=lightreid.data.build_train_dataset(['market1501', 'dukemtmtcreid'], conbimeall=False)
            target=lightreid.data.build_test_dataset('market1501')
            transforms_train=lightreid.data.build_transforms(img_size=[256, 128], transforms_list=['randomflip', 'padcrop', 'rea']),
            transforms_test=lightreid.data.build_transforms(img_size=[256, 128], transforms_list=[]),
            sampler='pk', p=16, k=4)
    """
    train_datasets = []
    for dataset_name in dataset_list:
        assert dataset_name in __datasets_factory.keys(), \
            'expect dataset in {}}, but got {}'.format(__datasets_factory.keys(), dataset_name)
        dataset = __datasets_factory[dataset_name](__datasets_path[dataset_name]['path'], combineall)
        train_datasets.append(dataset)
    return train_datasets


def build_test_dataset(dataset_name):
    '''
    Example:
        see build_train_dataset
    '''
    if isinstance(dataset_name, str):
        assert dataset_name in __datasets_factory.keys(), \
            'expect dataset in {}}, but got {}'.format(__datasets_factory.keys(), dataset_name)
        dataset = __datasets_factory[dataset_name](__datasets_path[dataset_name]['path'], combineall=False)
        return [dataset]
    elif isinstance(dataset_name, list):
        for val in dataset_name:
            assert val in __datasets_factory.keys(), \
                'expect dataset in {}}, but got {}'.format(__datasets_factory.keys(), val)
        dataset_list = []
        for val in dataset_name:
            dataset = __datasets_factory[val](__datasets_path[val]['path'], combineall=False)
            dataset_list.append(dataset)
        return dataset_list
    else:
        assert 0, 'expect input type string or list, but got {}'.format(type(dataset_name))

