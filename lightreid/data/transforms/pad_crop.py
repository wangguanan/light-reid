"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import torchvision.transforms as transforms

def padcrop(img_size):
    '''
    pad and crop
    Args:
        img_size(list): [height, width]
    '''
    return transforms.Compose([transforms.Pad(10), transforms.RandomCrop(img_size)])