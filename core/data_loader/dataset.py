import numpy as np
from PIL import Image
import copy
import os
import copy
from prettytable import PrettyTable

def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files


class PersonReIDSamples:

    def __init__(self, samples_path, reorder=True):

        # parameters
        self.samples_path = samples_path
        self.reorder = reorder

        # load samples
        samples = self._load_images_path(self.samples_path)

        # reorder person identities and camera identities
        if self.reorder:
            samples = self._reorder_labels(samples, 1)
            samples = self._reorder_labels(samples, 2)
        self.samples = samples


    def _reorder_labels(self, samples, label_index):

        ids = []
        for sample in samples:
            ids.append(sample[label_index])

        # delete repetitive elments and order
        ids = list(set(ids))
        ids.sort()
        # reorder
        for sample in samples:
            sample[label_index] = ids.index(sample[label_index])

        return samples


    def _load_images_path(self, folder_dir):
        '''
        :param folder_dir:
        :return: [(path, identiti_id, camera_id)]
        '''
        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_file_name(file_name)
                samples.append([root_path + file_name, identi_id, camera_id])
        return samples

    def _analysis_file_name(self, file_name):
        '''
        :param file_name: format like 0844_c3s2_107328_01.jpg
        :return:
        '''
        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id



class Samples4Market(PersonReIDSamples):
    '''
    Market Dataset
    '''
    pass


class Samples4Duke(PersonReIDSamples):
    '''
    Duke dataset
    '''
    def _analysis_file_name(self, file_name):
        '''
        :param file_name: format like 0002_c1_f0044158.jpg
        :return:
        '''
        split_list = file_name.replace('.jpg', '').replace('c', '').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id

class Samples4MSMT17(PersonReIDSamples):
    '''
    MSMT17 dataset
    '''
    def __init__(self, path, combineall=False):
        list_train_path = os.path.join(path, 'list_train.txt')
        list_val_path = os.path.join(path, 'list_val.txt')
        list_query_path = os.path.join(path, 'list_query.txt')
        list_gallery_path = os.path.join(path, 'list_gallery.txt')

        train = self._load_list(os.path.join(path, 'train/'), list_train_path)
        val = self._load_list(os.path.join(path, 'train/'), list_val_path)
        query = self._load_list(os.path.join(path, 'test/'), list_query_path)
        gallery = self._load_list(os.path.join(path, 'test/'), list_gallery_path)

        train += copy.deepcopy(val)
        if combineall:
            train += copy.deepcopy(query) + copy.deepcopy(gallery)
        train = self._reorder_labels(train, 1)
        self._analyze_samples(train, query, gallery)
        self.train, self.query, self.gallery = train, query, gallery


    def _load_list(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        data = []
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid) # no need to relabel
            camid = int(img_path.split('_')[2]) - 1 # index starts from 0
            img_path = os.path.join(dir_path, img_path)
            data.append([img_path, pid, camid])
        return data

    def _analyze_samples(self, train, query, gallery):

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
        table.add_row(['MSMT17', '', '', ''])
        table.add_row(['train', str(train_info[0]), str(train_info[1]), str(train_info[2])])
        table.add_row(['query', str(query_info[0]), str(query_info[1]), str(query_info[2])])
        table.add_row(['gallery', str(gallery_info[0]), str(gallery_info[1]), str(gallery_info[2])])
        print(table)


class PersonReIDDataSet:

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):

        this_sample = copy.deepcopy(self.samples[index])

        this_sample[0] = self._loader(this_sample[0])
        if self.transform is not None:
            this_sample[0] = self.transform(this_sample[0])
        this_sample[1] = np.array(this_sample[1])

        return this_sample

    def __len__(self):
        return len(self.samples)

    def _loader(self, img_path):
        return Image.open(img_path).convert('RGB')
