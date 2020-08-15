"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

from .backbones import resnet18, resnet34, resnet50, resnet101, resnet152
from .heads import BNHead, PCBHead, CodePyramid
from .layers import GeneralizedMeanPoolingP, Circle
from .models import BaseReIDModel