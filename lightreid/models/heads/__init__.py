from .bn_head import BNHead
from .code_pyramid import CodePyramid
from .pcb_head import PCBHead

__head_factory = {
    'bnhead': BNHead
}

def build_head(name, in_dim, class_num, classifier, **kwargs):
    assert name in __head_factory.keys(), 'expect a head in {} but got {}'.format(__head_factory.keys(), name)
    return __head_factory[name](in_dim=in_dim, class_num=class_num, classifier=classifier, **kwargs)