"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

from .vit_timm import *

__all__ = [
    'vit_small_patch16_224', 
    'vit_base_patch16_224', 
    'vit_base_patch32_224',
    'vit_base_patch16_384',
    'vit_base_patch32_384',
    'vit_large_patch16_224',
    'vit_large_patch32_224',
    'vit_large_patch16_384',
    'vit_large_patch32_384',
    'vit_base_patch16_224_in21k',
    'vit_base_patch32_224_in21k',
    'vit_large_patch16_224_in21k',
    'vit_large_patch32_224_in21k',
    'vit_huge_patch14_224_in21k',
    'vit_deit_tiny_patch16_224',
    'vit_deit_small_patch16_224',
    'vit_deit_base_patch16_224',
    'vit_deit_base_patch16_384',
    'vit_deit_tiny_distilled_patch16_224',
    'vit_deit_small_distilled_patch16_224',
    'vit_deit_base_distilled_patch16_224',
    'vit_deit_base_distilled_patch16_384',
    'vit_base_patch16_224_miil_in21k',
    'vit_base_patch16_224_miil',
]
