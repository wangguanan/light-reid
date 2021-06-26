from .architectures import ARCHs_REGISTRY, BaseReIDModel, ReductionReIDModel

from .backbones import build_cnnbackbone
from .layers import build_pooling
from .heads import build_head
from easydict import EasyDict as edict


__all__ = ['build_model']

def build_model(backbone, pooling, head, **kwargs):
    """
    Args:
    Example:
    """
    arch_factory__ = {
        'base_arch': BaseReIDModel,
        'reduction_arch': ReductionReIDModel,
        # user customized
        **ARCHs_REGISTRY._obj_map,
    }

    # convert type dict to easy dict for easy using
    backbone = edict(backbone)
    pooling = edict(pooling)
    head = edict(head)

    # init
    backbone = build_cnnbackbone(name=backbone.pop('name'), **backbone)
    pooling = build_pooling(**pooling)
    head = build_head(in_dim=backbone.dim, class_num=head.pop('class_num'), classifier=head.pop('classifier'), **head)

    if 'name' not in kwargs:
        return BaseReIDModel(backbone=backbone, pooling=pooling, head=head)
    else:
        kwargs = edict(kwargs)
        name = kwargs.pop('name')
        return arch_factory__[name](backbone=backbone, pooling=pooling, head=head, **kwargs)
