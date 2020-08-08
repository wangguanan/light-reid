"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

class Criterion(object):
    '''
    Example:
        criterion = Creterion([
            {'criterion': torch.nn.CrossEntropyLoss(), 'weight': 1.0},
            {'criterion': lightreid.losses.TripletLoss(margin=0.3, metric='cosine'), 'weight': 1.0},
        ])

        imgs, pids, camids = data
        feats, logits = model(imgs)
        loss = criterion.compute(feats=feats, logits=logits, pids=pids)
    '''

    CRITERION_FACTORY = [
        'CrossEntropyLoss', 'CrossEntropyLabelSmooth', 'TripletLoss', 'FocalLoss', 'CenterLoss',
        'ProbSelfDistillLoss', 'SIMSelfDistillLoss', 'KLLoss']
    VALUE_FACTORY = ['feats', 'head_feats', 'logits', 'pids', 'camids', 'feats_t', 'logits_t']

    def __init__(self, criterion_list):
        # check criterion class
        for criterion in criterion_list:
            assert criterion['criterion'].__class__.__name__ in Criterion.CRITERION_FACTORY, \
                'expect one of {}, but got {}'.format(Criterion.CRITERION_FACTORY, criterion['criterion'].__class__.__name__)
        self.criterion_list = criterion_list

    def compute(self, **kwargs):

        overall_loss = 0
        loss_dict = {}

        # check input is legal
        for arg in kwargs.keys():
            assert arg in Criterion.VALUE_FACTORY, \
                'expect one of {}, but got {}'.format(Criterion.VALUE_FACTORY, arg)

        # compute weighted loss
        for value in self.criterion_list:
            weight = value['weight']
            criterion = value['criterion']

            if criterion.__class__.__name__ in ['CrossEntropyLoss', 'CrossEntropyLabelSmooth', 'FocalLoss', 'CenterLoss']:
                if isinstance(kwargs['logits'], list): # for multi-head model (e.g. pcb), compute their average loss
                    loss = 0
                    for idx in range(len(kwargs['logits'])):
                        loss += weight * criterion(kwargs['logits'][idx], kwargs['pids'])
                    if 'reduce' in value.keys() and value['reduce'] == 'mean':
                        loss *= 1 / len(kwargs['logits'])

                else: # for single-head model (e,g, ide, bot)
                    loss = weight * criterion(kwargs['logits'], kwargs['pids'])

            elif criterion.__class__.__name__ in ['TripletLoss']:
                if isinstance(kwargs['feats'], list): # for multi-head model (e.g. pcb), compute their average loss
                    loss = 0
                    for idx in range(len(kwargs['feats'])):
                        loss += weight * criterion(kwargs['feats'][idx], kwargs['pids'])
                    if 'reduce' in value.keys() and value['reduce'] == 'mean':
                        loss *= 1 / len(kwargs['feats'])
                else: # for single-head model (e,g, ide, bot)
                    loss = weight * criterion(kwargs['feats'], kwargs['pids'])

            elif criterion.__class__.__name__ in ['SIMSelfDistillLoss']:
                assert 'head_feats' in kwargs.keys(), \
                    'SimDistillLoss expect feats as inputs, but got {}'.format(kwargs.keys())
                loss = weight * criterion(feats_list=kwargs['head_feats'])

            elif criterion.__class__.__name__ in ['ProbSelfDistillLoss']:
                assert 'logits' in kwargs.keys(), \
                    'SimDistillLoss expect feats as inputs, but got {}'.format(kwargs.keys())
                loss = weight * criterion(logits_list=kwargs['logits'])

            elif criterion.__class__.__name__ in ['KLLoss']:
                if isinstance(kwargs['logits'], list):
                    loss = weight * criterion(logits_s=kwargs['logits'][0], logits_t=kwargs['logits_t'].detach())
                else:
                    loss = weight * criterion(logits_s=kwargs['logits'], logits_t=kwargs['logits_t'].detach())

            else:
                assert 0, 'expect criterion in {} but got {}'.format(Criterion.CRITERION_FACTORY, criterion)


            overall_loss += loss
            loss_dict[criterion.__class__.__name__] = loss.data
            del loss

        return overall_loss, loss_dict