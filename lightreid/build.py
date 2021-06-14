from .data import build_datamanager
from .models import build_model
from .optim import build_optimizer
from .evaluations import build_evaluator
from .engine import Engine, Inference
from .losses import build_criterion
from easydict import EasyDict as edict


engine_factory__ = {
    'engine': Engine,
}


def build_engine(cfg):

    # cfg
    cfg = edict(cfg)

    # build datamanager
    datamanager = build_datamanager(**cfg.data)

    # build model
    cfg.model.head.class_num = datamanager.class_num
    model = build_model(**cfg.model)

    # build criterion
    cfg.criterion.num_classes = datamanager.class_num
    criterion = build_criterion(cfg.criterion)

    # build optim
    cfg.optim.optimizer.params = model.parameters()
    optim = build_optimizer(**cfg.optim)

    # build evaluator
    cfg.evaluator = cfg['evaluator'] if 'evaluator' in cfg.keys() else {
        'name': 'cmc_map_eval',
        'metric': 'cosine',
        'mode': 'inter-camera'
    }
    evaluator = build_evaluator(**cfg.evaluator)

    # build solver
    engine_type = cfg['engine'] if 'engine' in cfg.keys() else 'engine'
    solver = engine_factory__[engine_type](
        datamanager=datamanager, model=model, criterion=criterion, optimizer=optim,
        evaluator=evaluator,
        **cfg.env, **cfg.lightreid)

    return solver


def build_inference(cfg, model_path, use_gpu=False):

    # cfg
    cfg = edict(cfg)

    # build model
    cfg.model.head.class_num = 1 # classifier is never used during the inference stage
    model = build_model(**cfg.model)

    # build inference
    inference = Inference(model, img_size=cfg.data.pop('img_size'), model_path=model_path, use_gpu=use_gpu, **cfg.data, **cfg.lightreid)

    return inference
