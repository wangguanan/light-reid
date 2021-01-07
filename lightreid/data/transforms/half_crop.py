import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import cv2



class HalfCrop(object):
    """Half Crop Augmentation
    random crop bottom half of a pedestrian image (i.e. waist to foot)
    perform well for occluded reid
    Args:
        prob(float): probability to perform half crop
        keep_range(list): in height dimension, keep range
    """

    def __init__(self, prob=0.5,  keep_range=(0.50, 1.5)):
        self.prob = prob
        self.keep_range = keep_range

    def __call__(self, img):
        '''
        Args:
            img(np.array): image
        '''
        do_aug = random.uniform(0,1) < self.prob
        if do_aug:
            ratio = random.uniform(self.keep_range[0], self.keep_range[1])
            w, h = img.size
            tw = w
            th = int(h * ratio)
            img = F.crop(img, 0, 0, th, tw)
            img = F.resize(img, [h, w], Image.BILINEAR)
            return img
        else:
            return img
