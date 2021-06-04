import torch

from lightreid.losses import LOSSes_REGISTRY
from lightreid.losses import CrossEntropyLabelSmooth
from lightreid.losses import TripletLoss

__all__ = ['SkeletonIDELoss', 'SkeletonConstraintLoss', 'SkeletonGlobalTripletLoss']

@LOSSes_REGISTRY.register()
class SkeletonIDELoss:
    """
    Identity Discriminative Embedding Loss (IDE)
    Expect every local feature of skeleton is ID distinguishable
    Args:
        num_classes(int): class number
        weight_global(float): weight for global feature
    """

    def __init__(self, num_classes, weight_global):
        self.weight_global = weight_global
        self.ide_creiteron = CrossEntropyLabelSmooth(num_classes, reduce=False)

    def __call__(self, logits_list, pids, weights):
        """
        Args:
            logits_list(list): every element is a torch.tensor [bs, num_classes], the last one is the global feature
            pid(torch.tensor): [bs],
            weights(torch.tensor):  [bs, nums_local+nums_global], where nums_global always equal 1
        """
        loss_all = 0
        for i, feature_i in enumerate(logits_list):
            loss_i = self.ide_creiteron(feature_i, pids)
            if i == (len(logits_list) - 1):
                loss_all += (weights[:, i] * loss_i).mean()
            else:
                loss_all += (weights[:, i] * loss_i * self.weight_global).mean()
        return loss_all


@LOSSes_REGISTRY.register()
class SkeletonConstraintLoss:
    """
    Skeleton Constraint Loss
    Expect local features of symmetirc skeletons are equal,
        e.g. features from left and right shoulder are the same
    Args:
        symmetrics(list): every element is a tuple, e.g.
            [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
        weight_global(float):
    """

    def __init__(self, symmetrics, weight_global):
        self.symmetrics = symmetrics
        self.weight_global = weight_global

    def __call__(self, feat_list, weights):
        """
        Args:
            feat_list(list): every element is a torch.tensor with size [bs, dim],
                the last element shoud be the global feature
            weights(torch.tensor): [bs, nums_local+nums_global], where nums_global always equal 1
        """
        loss = 0
        for pos1, pos2 in self.symmetrics:
            diff = feat_list[pos1] - feat_list[pos2]
            distance = torch.pow(diff, 2).sum(1)
            confidence = (1 - torch.abs(weights[:, pos1] - weights[:, pos2])) * (
                    weights[:, pos1] + weights[:, pos2]) / 2
            confidence *= self.weight_global
            loss += (distance * confidence).mean()
        return loss


@LOSSes_REGISTRY.register()
class SkeletonGlobalTripletLoss:
    """
    """

    def __init__(self, margin, metric):
        self.criterion = TripletLoss(margin=margin, metric=metric, reduce=False)

    def __call__(self, feat_list, label, weights):
        """
        Args:
            feat_list(list): every element is a torch.tensor with size [bs, dim],
                the last element should be the global feature
        """
        emb = feat_list[-1]
        loss = self.criterion(emb, label)
        loss = (weights[:, -1] * loss).mean()
        return loss