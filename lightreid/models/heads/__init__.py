"""
Author: Guan'an Wang
E-mail: guan.wang0706@gmail.com
"""

from .build import HEADs_REGISTRY, build_head

# import heads, so they will be registered
from .bn_head import BNHead
from .code_pyramid import CodePyramid
from .pcb_head import PCBHead
