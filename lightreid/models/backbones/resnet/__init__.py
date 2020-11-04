"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

__all__ = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnet18ibna', 'resnet34ibna', 'resnet50ibna', 'resnet101ibn', 'resnet152ibna'
]

import torch.nn as nn
import torchvision
from .resnet_ibn import resnet18_ibn_a, resnet34_ibn_a, resnet50_ibn_a, resnet101_ibn_a, resnet152_ibn_a


class ResNet(nn.Module):

    def __init__(self, name, pretrained=True, last_stride_one=True):
        super(ResNet, self).__init__()

        self.name = name
        self.pretrained = pretrained
        self.last_stride_one = last_stride_one

        if self.name == 'resnet18':
            resnet = torchvision.models.resnet18(pretrained=pretrained)
            self.dim = 512
        elif self.name == 'resnet34':
            resnet = torchvision.models.resnet34(pretrained=pretrained)
            self.dim = 512
        elif self.name == 'resnet50':
            resnet = torchvision.models.resnet50(pretrained=pretrained)
            self.dim = 2048
        elif self.name == 'resnet101':
            resnet = torchvision.models.resnet101(pretrained=pretrained)
            self.dim = 2048
        elif self.name == 'resnet152':
            resnet = torchvision.models.resnet152(pretrained=pretrained)
            self.dim = 2048
        elif self.name == 'resnet18ibn':
            resnet = resnet18_ibn_a(pretrained=pretrained)
            self.dim = 512
        elif self.name == 'resnet34ibn':
            resnet = resnet34_ibn_a(pretrained=pretrained)
            self.dim = 512
        elif self.name == 'resnet50ibn':
            resnet = resnet50_ibn_a(pretrained=pretrained)
            self.dim = 2048
        elif self.name == 'resnet101ibn':
            resnet = resnet101_ibn_a(pretrained=pretrained)
            self.dim = 2048
        elif self.name == 'resnet152ibn':
            resnet = resnet152_ibn_a(pretrained=pretrained)
            self.dim = 2048
        else:
            assert 0, 'mode error, expect resnet18, resnet34, resnet50, resnet101, resnet152, but got {}'.format(self.name)

        if last_stride_one:
            for idx in range(len(resnet.layer4)):
                if name in ['resnet18', 'resnet34', 'resnet18ibn', 'resnet34ibn']:
                    resnet.layer4[idx].conv1.stride = (1, 1)
                else:
                    resnet.layer4[idx].conv2.stride = (1, 1)
                if resnet.layer4[idx].downsample is not None:
                    resnet.layer4[idx].downsample[0].stride = (1, 1)

        #
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18(pretrained=True, last_stride_one=True, **kwargs):
    return ResNet(name='resnet18', pretrained=pretrained, last_stride_one=last_stride_one)

def resnet34(pretrained=True, last_stride_one=True, **kwargs):
    return ResNet(name='resnet34', pretrained=pretrained, last_stride_one=last_stride_one)

def resnet50(pretrained=True, last_stride_one=True, **kwargs):
    return ResNet(name='resnet50', pretrained=pretrained, last_stride_one=last_stride_one)

def resnet101(pretrained=True, last_stride_one=True, **kwargs):
    return ResNet(name='resnet101', pretrained=pretrained, last_stride_one=last_stride_one)

def resnet152(pretrained=True, last_stride_one=True, **kwargs):
    return ResNet(name='resnet152', pretrained=pretrained, last_stride_one=last_stride_one)

def resnet18ibna(pretrained=True, last_stride_one=True, **kwargs):
    return ResNet(name='resnet18ibn', pretrained=pretrained, last_stride_one=last_stride_one)

def resnet34ibna(pretrained=True, last_stride_one=True, **kwargs):
    return ResNet(name='resnet34ibn', pretrained=pretrained, last_stride_one=last_stride_one)

def resnet50ibna(pretrained=True, last_stride_one=True, **kwargs):
    return ResNet(name='resnet50ibn', pretrained=pretrained, last_stride_one=last_stride_one)

def resnet101ibna(pretrained=True, last_stride_one=True, **kwargs):
    return ResNet(name='resnet101ibn', pretrained=pretrained, last_stride_one=last_stride_one)

def resnet152ibna(pretrained=True, last_stride_one=True, **kwargs):
    return ResNet(name='resnet152ibn', pretrained=pretrained, last_stride_one=last_stride_one)