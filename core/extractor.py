'''
given an or a list of image(s)
extrct its/their features
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import Base, DemoBase
from .model import Model
from tools import cosine_dist, euclidean_dist


def build_extractor(config, use_cuda):
    return Extractor(config.image_size, config.pid_num, config.model_path, use_cuda)

class Extractor(Base):
    '''
    given *RGB* image(s) in format of a list, each element is a numpy of size *[h,w,c]*, range *[0,225]*
    return their feature(s)(list), each element is a numpy of size [feat_dim]
    '''

    def __init__(self, image_size, pid_num, model_path, use_cuda):
        self.image_size = image_size
        self.pid_num = pid_num
        self.model_path = model_path
        self.use_cuda = use_cuda
        # init model
        self._init_device(use_cuda)
        self._init_model()
        # resume model
        self.resume_from_model(self.model_path)
        self.set_eval()

    def _init_device(self, use_cuda):
        if use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def _init_model(self):
        model = Model(class_num=self.pid_num)
        self.model = nn.DataParallel(model).to(self.device)

    def np2tensor(self, image):
        '''
        convert a numpy *hwc* image *(0,255)*  to a torch.tensor *chw* image *(0,1)*
        Args:
            image(numpy): [h,w,c], in format of RGB, range [0, 255]
        '''
        assert isinstance(image, np.ndarray), "input must be a numpy array!"
        image = image.astype(np.float) / 255.
        image = image.transpose([2,0,1])
        image = torch.from_numpy(image).float()
        return image

    def resize_images(self, images, image_size):
        '''resize a batch of images to image_size'''
        images = F.interpolate(images, image_size, mode='bilinear', align_corners=True)
        return images

    def normalize_images(self, images, mean=[0.485, 0.456, 0.406], std=[0.485, 0.456, 0.406]):
        '''
        Args:
            images(torch.tensor): size [bs, c, h, w], range [0,1]
        Return:
            images(torch.tensor): size [bs, c, h, w],
        '''
        mean = torch.tensor(mean).view([1, 3, 1, 1]).repeat([images.size(0), 1, images.size(2), images.size(3)]).to(self.device)
        std = torch.tensor(std).view([1, 3, 1, 1]).repeat([images.size(0), 1, images.size(2), images.size(3)]).to(self.device)
        images = (images - mean) / std
        return images

    def extract_list(self, image_list):
        '''
        given *RGB* image(s) in format of a list, each element is a numpy of size *[h,w,c]*, range *[0,225]*
        return their feature(s)(list), each element is a numpy of size [feat_dim]
        Args:
            image_list(list): every element is a numpy of size *[h,w,c]* format *RGB* and range *[0,255]*
        Return:
            feature_list(list): every element is a numpy of size [feature_dim]
        '''
        images = [self.resize_images(self.np2tensor(image).unsqueeze(0), self.image_size) for image in image_list]
        images = torch.cat(images, dim=0)
        images = images.to(self.device)
        images = self.normalize_images(images)
        with torch.no_grad():
            features = self.model(images)
        features = features.data.cpu().numpy()
        feature_list = [feature for feature in features]
        return feature_list

    # def extract_image(self, image):
    #     '''
    #     given an image, return its feature
    #     Args:
    #         image(torch.tensor): [c,h,w]
    #     Return:
    #         feature: [feature_dim]
    #     '''
    #     image = image.to(self.device)
    #     images = image.unsqueeze(0)
    #     images = self.resize_images(images, self.image_size)
    #     images = self.normalize_images(images)
    #     features = self.model(images)
    #     feature = features.squeeze(0)
    #     return feature
    #
    # def extract_images(self, images):
    #     '''
    #     given more than one image, return their feature
    #     Args:
    #         image(torch.tensor): [bs, c,h,w]
    #     Return:
    #         feature: [bs, feature_dim]
    #     '''
    #     images = images.to(self.device)
    #     images = self.resize_images(images, self.image_size)
    #     images = self.normalize_images(images)
    #     features = self.model(images)
    #     return features
