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


    def __init__(self, criterion_list):
        self.criterion_list = criterion_list

    def compute(self, **kwargs):

        overall_loss = 0
        loss_dict = {}

        # compute weighted loss
        for value in self.criterion_list:
            weight = value['weight']
            criterion = value['criterion']
            inputs_name = value['inputs']
            if isinstance(inputs_name, str): # (x)
                loss = weight * criterion(kwargs[inputs_name])
            elif isinstance(inputs_name, dict): # (x, y)
                inputs_tmp = {}
                for key, val in inputs_name.items():
                    inputs_tmp[key] = kwargs[val]
                loss = weight * criterion(**inputs_tmp)
            else:
                raise RuntimeError('type error')

            overall_loss += loss
            name = criterion.__class__.__name__ if 'display_name' not in value.keys() else value['display_name']
            loss_dict[name] = loss.data
            del loss
            value, weight, criterion, inputs_name = None, None, None, None

        return overall_loss, loss_dict


