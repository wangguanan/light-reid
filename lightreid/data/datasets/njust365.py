"""
@author:    wangguanan
@contact:   guan.wang0706@gmail.com
"""

import os, copy
from os.path import join, realpath, dirname
from easydict import EasyDict

from .reid_samples import ReIDSamples
from lightreid.utils import os_walk

__all__ = ['NJUST365', 'NJUST365SPR', 'NJUST365WIN']


class NJUST365(ReIDSamples):
    '''
    njust365-all
    njust365 is a private dataset, not released yet
    '''

    def __init__(self, data_path, combineall=False, season='all', **kwargs):
        assert season in ['win', 'spr', 'all'], \
            'expect season in [win, spr, all], but got {}'.format(season)

        # init winter data
        if season in ['win', 'all']:
            win = EasyDict()
            train_path = join(data_path, 'copy_dataset_win_train_backup/')
            query_path = join(data_path, 'copy_dataset_win_test_query/')
            gallery_path = join(data_path, 'copy_dataset_win_test_gallery/')
            train = self._load_images_path(train_path, idstartfrom=0)
            query = self._load_querygallery_images_path(query_path, idstartfrom=0)
            gallery = self._load_querygallery_images_path(gallery_path, idstartfrom=0)
            win.train, win.query, win.gallery = \
                copy.deepcopy(train), copy.deepcopy(query), copy.deepcopy(gallery)
            del train, query, gallery

        # init spring data
        if season in ['spr', 'all']:
            spr = EasyDict()
            train_path = join(data_path, 'copy_dataset_spr_train/')
            query_path = join(data_path, 'copy_dataset_spr_test_query/')
            gallery_path = join(data_path, 'copy_dataset_spr_test_gallery/')
            # different peoples in spring and winter may use same ids, thus we add a very larege bias to fix the bug
            train = self._load_images_path(train_path, idstartfrom=999999)
            query = self._load_querygallery_images_path(query_path, idstartfrom=999999)
            gallery = self._load_querygallery_images_path(gallery_path, idstartfrom=999999)
            spr.train, spr.query, spr.gallery = \
                copy.deepcopy(train), copy.deepcopy(query), copy.deepcopy(gallery)
            del train, query, gallery

        # return final results
        if season == 'win':
            super(NJUST365, self).__init__(win.train, win.query, win.gallery, combineall)
        elif season == 'spr':
            super(NJUST365, self).__init__(spr.train, spr.query, spr.gallery, combineall)
        else:
            train = self.combine([copy.deepcopy(win.train), copy.deepcopy(spr.train)])
            query = self.combine([copy.deepcopy(win.query), copy.deepcopy(spr.query)])
            gallery = self.combine([copy.deepcopy(win.gallery), copy.deepcopy(spr.gallery)])
            super(NJUST365, self).__init__(train, query, gallery, combineall)

    def _load_images_path(self, path, idstartfrom=0):
        txt_path = join(path, 'list.txt')
        samples = []
        if os.path.exists(txt_path):
            txt_file = open(txt_path, 'r')
            lines = txt_file.readlines()
            for line in lines:
                img_path, pid, cid = line.split(',')
                pid = int(pid)
                cid = int(cid)
                samples.append([img_path, pid + idstartfrom, cid + idstartfrom])
            txt_file.close()
        else:
            txt_file = open(txt_path, 'w')
            _, folders, _ = os_walk(path)
            samples = []
            for folder in folders:
                _, _, files = os_walk(os.path.join(path, folder))
                for file in files:
                    if '.jpg' in file or '.png' in file:
                        pid = int(folder)
                        try:
                            cid = int(file[5:7])
                        except:
                            continue
                        txt_file.writelines('{path},{pid},{cid}\n'.format(path=join(join(path, folder), file), pid=pid, cid=cid))
                        samples.append([os.path.join(os.path.join(path, folder), file), pid+idstartfrom, cid+idstartfrom])
            txt_file.close()
        return samples

    def _load_querygallery_images_path(self, path, idstartfrom=0):
        txt_path = join(path, 'list.txt')
        samples = []
        if os.path.exists(txt_path):
            txt_file = open(txt_path, 'r')
            lines = txt_file.readlines()
            for line in lines:
                img_path, pid, cid = line.split(',')
                pid = int(pid)
                cid = int(cid)
                samples.append([img_path, pid + idstartfrom, cid + idstartfrom])
            txt_file.close()
        else:
            txt_file = open(txt_path, 'w')
            _, _, files = os_walk(path)
            for file in files:
                if '.jpg' in file:
                    pid = int(file.split('_')[0])
                    cid = int(file.split('_')[1][5:7])
                    txt_file.writelines('{path},{pid},{cid}\n'.format(path=join(path, file), pid=pid, cid=cid))
                    samples.append([join(path, file), pid+idstartfrom, cid+idstartfrom])
            txt_file.close()
        return samples

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


class NJUST365WIN(NJUST365):
    """
    njust365-winter dataset
    """
    def __init__(self, data_path, combineall, **kwargs):
        super(NJUST365WIN, self).__init__(data_path, combineall, 'win', **kwargs)

class NJUST365SPR(NJUST365):
    '''
    njust365-spring dataset
    '''
    def __init__(self, data_path, combineall, **kwargs):
        super(NJUST365SPR, self).__init__(data_path, combineall, 'spr', **kwargs)
