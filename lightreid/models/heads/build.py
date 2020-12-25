from lightreid.utils import Registry

HEADs_REGISTRY = Registry('heads')

def build_head(name, in_dim, class_num, classifier, **kwargs):
    __head_factory = {
        # original support
        'bnhead': HEADs_REGISTRY.get('BNHead'),
        # user customized
        **HEADs_REGISTRY._obj_map
    }
    assert name in __head_factory.keys(), 'expect a head in {} but got {}'.format(__head_factory.keys(), name)
    return __head_factory[name](in_dim=in_dim, class_num=class_num, classifier=classifier, **kwargs)