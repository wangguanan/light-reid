from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .resnet import resnet18ibna, resnet34ibna, resnet50ibna, resnet101ibna, resnet152ibna


__cnnbackbone_factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet18ibna': resnet18ibna,
    'resnet34ibna': resnet34ibna,
    'resnet50ibna': resnet50ibna,
    'resnet101ibna': resnet101ibna,
    'resnet152ibna': resnet152ibna,
}


def build_cnnbackbone(name, pretrained, last_stride_one, **kwargs):
    return __cnnbackbone_factory[name](pretrained=pretrained, last_stride_one=last_stride_one, **kwargs)