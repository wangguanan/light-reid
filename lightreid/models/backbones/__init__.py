from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .resnet import resnet18ibna, resnet34ibna, resnet50ibna, resnet101ibna, resnet152ibna
from .transformers import *

__cnnbackbone_factory = {
    # resnet series
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
    # vision transformer series
    'vit_small_patch16_224': vit_small_patch16_224,
    'vit_base_patch16_224': vit_base_patch16_224,
    'vit_base_patch32_224': vit_base_patch32_224,
    'vit_base_patch16_384': vit_base_patch16_384,
    'vit_base_patch32_384': vit_base_patch32_384,
    'vit_large_patch16_224': vit_large_patch16_224,
    'vit_large_patch32_224': vit_large_patch32_224,
    'vit_large_patch16_384': vit_large_patch16_384,
    'vit_large_patch32_384': vit_large_patch32_384,
    'vit_base_patch16_224_in21k': vit_base_patch16_224_in21k,
    'vit_base_patch32_224_in21k': vit_base_patch32_224_in21k,
    'vit_large_patch16_224_in21k': vit_large_patch16_224_in21k,
    'vit_large_patch32_224_in21k': vit_large_patch32_224_in21k,
    'vit_huge_patch14_224_in21k': vit_huge_patch14_224_in21k,
    'vit_deit_tiny_patch16_224': vit_deit_tiny_patch16_224,
    'vit_deit_small_patch16_224': vit_deit_small_patch16_224,
    'vit_deit_base_patch16_224': vit_deit_base_patch16_224,
    'vit_deit_base_patch16_384': vit_deit_base_patch16_384,
    'vit_deit_tiny_distilled_patch16_224': vit_deit_tiny_distilled_patch16_224,
    'vit_deit_small_distilled_patch16_224': vit_deit_small_distilled_patch16_224,
    'vit_deit_base_distilled_patch16_224': vit_deit_base_distilled_patch16_224,
    'vit_deit_base_distilled_patch16_384': vit_deit_base_distilled_patch16_384,
    'vit_base_patch16_224_miil_in21k': vit_base_patch16_224_miil_in21k,
    'vit_base_patch16_224_miil': vit_base_patch16_224_miil,
}


def build_cnnbackbone(name, pretrained=True, **kwargs):
    return __cnnbackbone_factory[name](pretrained=pretrained, **kwargs)