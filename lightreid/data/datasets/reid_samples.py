"""
@author:    wangguanan
@contact:   guan.wang0706@gmail.com
"""

import os
import os.path as osp
from prettytable import PrettyTable
import tarfile
import zipfile
import time
import sys
import errno
import copy


class ReIDSamples:
    '''
    An abstract class representing a Re-ID samples.
    Attrs:
        train (list): contains tuples of (img_path(s), pid, camid).
        query (list): contains tuples of (img_path(s), pid, camid).
        gallery (list): contains tuples of (img_path(s), pid, camid).
    '''

    def __init__(self, train, query, gallery, combineall=False, **kwargs):

        if combineall:
            print('[{} Combine All] combine train, query and gallery and training set ... ...'.format(self.__class__.__name__))
            train += copy.deepcopy(query) + copy.deepcopy(gallery)
        if train is not None:
            train = self.relabel(train)
        self.train, self.query, self.gallery = train, query, gallery

        # show information
        self.statistics(train=self.train, query=self.query, gallery=self.gallery, combineall=combineall)


    def statistics(self, **kwargs):
        '''show sample statistics'''
        def analyze(samples):
            if samples is None:
                return None, None, None
            pid_num = len(set([sample[1] for sample in samples]))
            cid_num = len(set([sample[2] for sample in samples]))
            sample_num = len(samples)
            return sample_num, pid_num, cid_num

        table = PrettyTable([self.__class__.__name__, 'images', 'identities', 'cameras', 'imgs/id', 'imgs/cam', 'imgs/id&cam'])
        for key, val in kwargs.items():
            if key in ['train', 'query', 'gallery']:
                info = analyze(val)
                key_str = str(key)
                if 'combineall' in kwargs.keys() and kwargs['combineall'] and key == 'train':
                    key_str += '(combineall)'
                img_num, pid_num, cid_num = info
                imgs_per_id = round(img_num / float(pid_num), 2) if img_num is not None else None
                imgs_per_cam = round(img_num / float(cid_num), 2) if img_num is not None else None
                imgs_per_idcam = round(img_num / float(pid_num) / float(cid_num), 2) if img_num is not None else None
                table.add_row([str(key_str), str(info[0]), str(info[1]), str(info[2]),
                               str(imgs_per_id), str(imgs_per_cam), str(imgs_per_idcam)])
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

    def download_dataset(self, dataset_dir, dataset_url):
        """Downloads and extracts dataset.

        Args:
            dataset_dir (str): dataset directory.
            dataset_url (str): url to download dataset.
        """
        if osp.exists(dataset_dir):
            return

        if dataset_url is None:
            raise RuntimeError(
                '{} dataset needs to be manually '
                'prepared, please follow the '
                'document to prepare this dataset'.format(
                    self.__class__.__name__
                )
            )

        print('Creating directory "{}"'.format(dataset_dir))
        self.mkdir_if_missing(dataset_dir)
        fpath = osp.join(dataset_dir, osp.basename(dataset_url))

        print(
            'Downloading {} dataset to "{}"'.format(
                self.__class__.__name__, dataset_dir
            )
        )
        self.download_url(dataset_url, fpath)

        print('Extracting "{}"'.format(fpath))
        try:
            tar = tarfile.open(fpath)
            tar.extractall(path=dataset_dir)
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(dataset_dir)
            zip_ref.close()

        print('{} dataset is ready'.format(self.__class__.__name__))


    def download_url(self, url, dst):
        """Downloads file from a url to a destination.

        Args:
            url (str): url to download file.
            dst (str): destination path.
        """
        from six.moves import urllib
        print('* url="{}"'.format(url))
        print('* destination="{}"'.format(dst))

        def _reporthook(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024*duration))
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(
                '\r...%d%%, %d MB, %d KB/s, %d seconds passed' %
                (percent, progress_size / (1024*1024), speed, duration)
            )
            sys.stdout.flush()

        urllib.request.urlretrieve(url, dst, _reporthook)
        sys.stdout.write('\n')


    def mkdir_if_missing(self, dirname):
        """Creates dirname if it is missing."""
        if not osp.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))