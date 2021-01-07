import torch
import torch.nn as nn

from lightreid.models.architectures import ARCHs_REGISTRY

__all__ = ['SkeletonReIDModel']

@ARCHs_REGISTRY.register()
class SkeletonReIDModel(nn.Module):
    """
    Architecture for ReID Model (using skeleton)
        combine backbone, pooling and head modules
        remove pooling, i.e. img --> backbone --> feature map --> head
    """

    def __init__(self, backbone, pooling, head):
        super(SkeletonReIDModel, self).__init__()
        self.backbone = backbone
        self.pooling = pooling
        assert head.__class__.__name__ == 'SkeletonMultiHeads', \
            'SkeletonReIDModel only support SkeletonMultiHeads but got {}'.format(head.__class__.__name__)
        self.head = head
        self.disable_hash()


    def forward(self, x, y=None, fixcnn=False):
        '''
        Args:
            x(torch.tensor): images
            y(torch.tensor): labels, required by circle_softmax, arc_softmax
            fixcnn(bool): if True, detach feature map
        '''
        # cnn backbone
        feats_map = self.backbone(x)
        if fixcnn:
            feats_map = feats_map.detach()
        ## pooling
        # feats_vec = self.pooling(feats_map).squeeze(3).squeeze(2) # support bs=1
        # head
        res = self.head(x, feats_map, y)

        # return
        res['feats_map'] = feats_map
        # res['feats_vec'] = feats_vec

        if self.training:
            return res
        else:
            return res[self.test_feats]

    def enable_hash(self):
        self.head.enable_hash()
        self.test_feats = 'binary_feats'

    def disable_hash(self):
        self.head.disable_hash()
        self.test_feats = 'bn_feats'



