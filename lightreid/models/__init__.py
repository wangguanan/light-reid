"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

from .backbones import resnet18, resnet34, resnet50, resnet101, resnet152
from .heads import BNHead, PCBHead, CodePyramid
from .layers import GeneralizedMeanPoolingP, Circle, ArcFace
from .architectures import BaseReIDModel, ReductionReIDModel

from .backbones import build_cnnbackbone
from .layers import build_pooling, build_classifier
from .heads import build_head
from easydict import EasyDict as edict


arch_factory__ = {
    'base_arch': BaseReIDModel,
    'reduction_arch': ReductionReIDModel
}

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
    backbone = build_cnnbackbone(name=backbone.name, pretrained=backbone.pretrained, last_stride_one=backbone.last_stride_one)
    pooling = build_pooling(**pooling)
    classifier_indim = backbone.dim if 'middle_dim' not in head.keys() else head.middle_dim
    classifier = build_classifier(name=head.classifier.pop('name'), in_dim=classifier_indim, out_dim=head.class_num, **head.pop('classifier'))
    head = build_head(in_dim=backbone.dim, class_num=head.pop('class_num'), classifier=classifier, **head)

    if 'name' not in kwargs:
        return BaseReIDModel(backbone=backbone, pooling=pooling, head=head)
    else:
        kwargs = edict(kwargs)
        name = kwargs.pop('name')
        return arch_factory__[name](backbone=backbone, pooling=pooling, head=head, **kwargs)


