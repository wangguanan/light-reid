"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

from .backbones import resnet18, resnet34, resnet50, resnet101, resnet152
from .heads import BNHead, PCBHead, CodePyramid
from .layers import GeneralizedMeanPoolingP, Circle, ArcFace
from .models import BaseReIDModel

from .backbones import build_cnnbackbone
from .layers import build_pooling, build_classifier
from .heads import build_head
from easydict import EasyDict as edict


def build_model_wihcfg(config):
    return build_model(*config)


def build_model(backbone, pooling, head, **kwargs):
    """
    Args:
    Example:
    """

    # convert type dict to easy dict for easy using
    backbone = edict(backbone)
    pooling = edict(pooling)
    head = edict(head)

    # init
    backbone = build_cnnbackbone(name=backbone.name, pretrained=backbone.pretrained,
                                 last_stride_one=backbone.last_stride_one)
    pooling = build_pooling(name=pooling.name)
    classifier = build_classifier(name=head.classifier.pop('name'), in_dim=backbone.dim, out_dim=head.class_num,
                                  **head.classifier)
    head = build_head(name=head.name, in_dim=backbone.dim, class_num=head.class_num, classifier=classifier)

    return BaseReIDModel(backbone=backbone, pooling=pooling, head=head)
