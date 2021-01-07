from .market1501 import Market1501
from .dukemtmcreid import DukeMTMCreID
from .msmt17 import MSMT17
from .cuhk03 import CUHK03
from .wildtrack import WildTrackCrop
from .rap import RAP
from .njust365 import NJUST365, NJUST365SPR, NJUST365WIN
from .airportalert import AirportAlert
from .prid import PRID
from .occluded_reid import OccludedReID
from .partial_ilids import PartialILIDS
from .partial_reid import PartialReID

import yaml
from os.path import realpath, dirname, join


__all__ = [
    'build_train_dataset', 'build_test_dataset']


__datasets_factory = {
    'market1501': Market1501,
    'dukemtmcreid': DukeMTMCreID,
    'msmt17': MSMT17,
    'cuhk03': CUHK03,
    'wildtrack_crop': WildTrackCrop,
    'rap': RAP,
    'njust365': NJUST365,
    'njust365spr': NJUST365SPR,
    'njust365win': NJUST365WIN,
    'airportalert': AirportAlert,
    'prid': PRID,
    'occludedreid': OccludedReID,
    'partialreid': PartialReID,
    'partialilids': PartialILIDS,
}


# init dataset paths
with open(join(dirname(__file__), 'datasetpaths.yaml')) as file:
    __datasets_config = yaml.load(file, Loader=yaml.FullLoader)


def build_train_dataset(dataset_list, combineall=False):
    """
    Args:
        dataset_list(list): a list of dataset name(str)
        combineall(bool): combine train and test set as train set
    Example:
        datamanager = lightreid.data.DataManager(
            sources=lightreid.data.build_train_dataset(['market1501', 'dukemtmtcreid'], conbimeall=False)
            target=lightreid.data.build_test_dataset('market1501')
            transforms_train=lightreid.data.build_transforms(img_size=[256, 128], transforms_list=['randomflip', 'padcrop', 'rea']),
            transforms_test=lightreid.data.build_transforms(img_size=[256, 128], transforms_list=[]),
            sampler='pk', p=16, k=4)
    """

    print("""
    building training datasets ... ... 
    """)
    train_datasets = []
    for dataset_name in dataset_list:
        assert dataset_name in __datasets_factory.keys(), \
            'expect dataset in {}, but got {}'.format(__datasets_factory.keys(), dataset_name)
        dataset_folder = join(__datasets_config[dataset_name]['path'], __datasets_config[dataset_name]['folder'])
        download = __datasets_config[dataset_name]['download']
        dataset = __datasets_factory[dataset_name](dataset_folder, combineall=combineall, download=download)
        train_datasets.append(dataset)
    return train_datasets


def build_test_dataset(dataset_name):
    '''
    Args:
        dataset_name: accept
            str: a dataset name
            list: a list of dataset names
    Example:
        see build_train_dataset
    '''

    print("""
    building test datasets ... ... 
    """)
    if isinstance(dataset_name, str):
        assert dataset_name in __datasets_factory.keys(), \
            'expect dataset in {}}, but got {}'.format(__datasets_factory.keys(), dataset_name)
        dataset_folder = join(__datasets_config[dataset_name]['path'], __datasets_config[dataset_name]['folder'])
        download = __datasets_config[dataset_name]['download']
        dataset = __datasets_factory[dataset_name](dataset_folder, combineall=False, download=download)
        return [dataset]
    elif isinstance(dataset_name, list):
        for val in dataset_name:
            assert val in __datasets_factory.keys(), \
                'expect dataset in {}}, but got {}'.format(__datasets_factory.keys(), val)
        dataset_list = []
        for val in dataset_name:
            dataset_folder = join(__datasets_config[val]['path'], __datasets_config[val]['folder'])
            download = __datasets_config[val]['download']
            dataset = __datasets_factory[val](dataset_folder, combineall=False, download=download)
            dataset_list.append(dataset)
        return dataset_list
    else:
        assert 0, 'expect input type string or list, but got {}'.format(type(dataset_name))

