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