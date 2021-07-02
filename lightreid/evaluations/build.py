from lightreid.utils import Registry

EVALUATORs_REGISTRY = Registry('evaluator')


def build_evaluator(name='cmc_map_eval', metric='cosine', mode='inter-camera', **kwargs):

    evaluator_factory_ = {
        # original support
        'cmc_map_eval': EVALUATORs_REGISTRY.get('CmcMapEvaluator'),
        'pre_recall_eval': EVALUATORs_REGISTRY.get('PreRecEvaluator'),
        'cmc_map_eval_1b1': EVALUATORs_REGISTRY.get('CmcMapEvaluator1b1'),
        'cmc_map_eval_c2f': EVALUATORs_REGISTRY.get('CmcMapEvaluatorC2F'),
        # user customized
        **EVALUATORs_REGISTRY._obj_map
    }

    return evaluator_factory_[name](metric=metric, mode=mode, **kwargs)
