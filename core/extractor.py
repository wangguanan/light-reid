'''
given an or a list of image(s)
extrct its/their features
'''

import torch
import torch.nn.functional as F
from .base import Base, DemoBase
from tools import cosine_dist, euclidean_dist


class Extractor(Base):

    def __init__(self, image_size, pid_num, model_path):
        self.image_size = image_size
        self.pid_num = pid_num
        self.model_path = model_path
        # init model
        self._init_device()
        self._init_model()
        # resume model
        self.resume_from_model(self.model_path)
        self.set_eval()


    def resize_images(self, images, image_size):
        '''resize a batch of images to image_size'''
        images = F.interpolate(images, image_size, mode='linear')
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

    def extract_image(self, image):
        '''
        given an image, return its feature
        Args:
            image(torch.tensor): [c,h,w]
        Return:
            feature: [feature_dim]
        '''
        image = image.to(self.device)
        images = image.unsqueeze(0)
        images = self.resize_images(images, self.image_size)
        images = self.normalize_images(images)
        features = self.model(images)
        feature = features.squeeze(0)
        return feature

    def extract_images(self, images):
        '''
        given more than one image, return their feature
        Args:
            image(torch.tensor): [bs, c,h,w]
        Return:
            feature: [bs, feature_dim]
        '''
        images = images.to(self.device)
        images = self.resize_images(images, self.image_size)
        images = self.normalize_images(images)
        features = self.model(images)
        return features

    def extract_list(self, image_list):
        '''
        given more than one image in format of a list, return their feature
        Args:
            image_list(list): every element is a torch.tensor of size [c, h, w]
        Return:
            feature_list(list): every element is a troch.tensor of size [feature_dim]
        '''
        images = [image.unsqueeze(0) for image in image_list]
        images = torch.cat(images, dim=0)
        images = images.to(self.device)
        images = self.resize_images(images, self.image_size)
        images = self.normalize_images(images)
        features = self.model(images)
        feature_list = [feature for feature in features]
        return feature_list
