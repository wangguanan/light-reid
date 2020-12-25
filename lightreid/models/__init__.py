"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

from .backbones import resnet18, resnet34, resnet50, resnet101, resnet152
from .heads import BNHead, PCBHead, CodePyramid
from .layers import GeneralizedMeanPoolingP, Circle, ArcFace
from .architectures import ARCHs_REGISTRY, BaseReIDModel, ReductionReIDModel

from .backbones import build_cnnbackbone
from .layers import build_pooling, build_classifier
from .heads import build_head

from .build import build_model



