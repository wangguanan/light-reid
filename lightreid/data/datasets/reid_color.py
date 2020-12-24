import numpy as np
import scipy.io as scio


class ReIDClothingColors4Market:
    '''
    Market-1501 clothing colors
    '''

    def __init__(self, attribute_file, upper=True):

        self.attribute_names_train = ('age', 'backpack', 'bag', 'handbag',
                                     'downblack', 'downblue', 'downbrown', 'downgray', 'downgreen', 'downpink', 'downpurple', 'downwhite', 'downyellow',
                                     'upblack', 'upblue', 'upgreen', 'upgray', 'uppurple', 'upred', 'upwhite', 'upyellow', 'clothes',
                                     'down', 'up', 'hair', 'hat', 'gender')
        self.attribute_names_query = ['age', 'backpack', 'bag', 'handbag', 'clothes', 'down', 'up', 'hair', 'hat', 'gender',
                                      'upblack', 'upwhite', 'upred', 'uppurple', 'upyellow', 'upgray', 'upblue', 'upgreen',
                                      'downblack', 'downwhite', 'downpink', 'downpurple', 'downyellow', 'downgray', 'downblue', 'downgreen', 'downbrown']

        if upper:
            self.clothing_color_names = ['upblack', 'upblue', 'upgreen', 'upgray', 'uppurple', 'upred', 'upwhite', 'upyellow']
        else:
            self.clothing_color_names = ['downblack', 'downblue', 'downbrown', 'downgray', 'downgreen', 'downpink', 'downpurple', 'downwhite', 'downyellow']

        self.attributes_train, _ = self._load_attribute_mat(attribute_file)
        self.clothing_colors_train = self._select_attributes(self.attributes_train, self.clothing_color_names, self.attribute_names_train)


    def _select_attributes(self, attributes, selected_attribute_names, all_attribute_names):
        '''
        select expect attributes from all attributes
        :param attrs: [identity_size, attribute_size]
        :param selected_attrs_name:
        :param all_attrs_name:
        :return: one hot
        '''
        selected_attrs = np.zeros([attributes.shape[0], len(selected_attribute_names)])
        for i, attr_name in enumerate(selected_attribute_names):
            index = all_attribute_names.index(attr_name)
            selected_attrs[:, i] = attributes[:, index]

        # if not labeled, return -1
        selected_attrs = np.concatenate([np.ones([selected_attrs.shape[0], 1]), selected_attrs], axis=1)
        selected_attrs_one_hot = np.argmax(selected_attrs, axis=1) - 1

        return selected_attrs_one_hot


    def _load_attribute_mat(self, attr_file):
        '''
        load market1501 attribute .mat file
        :param attr_file:
        :return: size [samples_num, attributes_num],
        '''

        data = scio.loadmat(attr_file)
        data = data['market_attribute']

        train = 1
        query = 0

        attr_train = np.zeros([27, 751])
        for i in xrange(27):
            for j in xrange(751):
                attr_train[i][j] = data[0][0][train][0][0][i][0][j]
        attr_query = np.zeros([27, 750])
        for i in xrange(27):
            for j in xrange(750):
                attr_query[i][j] = data[0][0][query][0][0][i][0][j]

        attr_train = np.transpose(attr_train)
        attr_query = np.transpose(attr_query)

        return attr_train, attr_query



class ReIDClothingColors4Duke(ReIDClothingColors4Market):


    def __init__(self, attribute_file, upper=True):

        self.attribute_names_train = ['backpack', 'bag', 'handbag', 'boots', 'gender', 'hat', 'shoes', 'top', 'downblack', 'downwhite', 'downred', 'downgray', 'downblue', 'downgreen', 'downbrown', 'upblack', 'upwhite', 'upred', 'uppurple', 'upgray', 'upblue', 'upgreen', 'upbrown']
        self.attribute_names_query = ['boots', 'shoes', 'top', 'gender', 'hat', 'backpack', 'bag', 'handbag', 'downblack', 'downwhite', 'downred', 'downgray', 'downblue', 'downgreen', 'downbrown', 'upblack', 'upwhite', 'upred', 'upgray', 'upblue', 'upgreen', 'uppurple', 'upbrown']

        if upper:
            self.clothing_color_names = ['upblack', 'upwhite', 'upred', 'uppurple', 'upgray', 'upblue', 'upgreen', 'upbrown']
        else:
            self.clothing_color_names = ['downblack', 'downwhite', 'downred', 'downgray', 'downblue', 'downgreen', 'downbrown']

        self.attributes_train, _ = self._load_attribute_mat(attribute_file)
        self.clothing_colors_train = self._select_attributes(self.attributes_train, self.clothing_color_names, self.attribute_names_train)


    def _load_attribute_mat(self, attr_file):
        '''
        load market1501 attribute .mat file
        :param attr_file:
        :return: size [samples_num, attributes_num]
        '''

        attr_num = 23
        p_num_train = 702
        p_num_test = 1110
        data = scio.loadmat(attr_file)
        data = data['duke_attribute']

        train = 0
        query = 1

        attr_train = np.zeros([attr_num, p_num_train])
        for i in xrange(attr_num):
            for j in xrange(p_num_train):
                attr_train[i][j] = data[0][0][train][0][0][i][0][j]
        attr_query = np.zeros([attr_num, p_num_test])
        for i in xrange(attr_num):
            for j in xrange(p_num_test):
                attr_query[i][j] = data[0][0][query][0][0][i][0][j]

        # sub 1 is important, map 1/2 to 0/1 for classification loss !!!
        attr_train = np.transpose(attr_train)
        attr_query = np.transpose(attr_query)

        return attr_train, attr_query