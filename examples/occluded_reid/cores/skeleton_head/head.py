"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .skeleton_model import SkeletonScoremap, pose_config, get_pose_net

from lightreid.models.heads import HEADs_REGISTRY
from lightreid.models.heads import build_head

__all__ = ['SkeletonMultiHeads']

@HEADs_REGISTRY.register()
class SkeletonMultiHeads(nn.Module):
    """
    Skeleton Multiple Heads, includes a
        skeleton local features, global features
    """

    def __init__(self, in_dim, class_num, classifier={'name': 'linear'}, middle_dim=None,
                 weight_global=1.0, pose_model_path='', test_mode='sum' # skeleton parameters
                 ):
        super(SkeletonMultiHeads, self).__init__()

        self.register_buffer('is_hash', torch.tensor(0).to(torch.bool))
        self.weight_global = weight_global
        self.test_mode = test_mode

        # init skeleton model
        self.skeleton_model = get_pose_net(pose_config, False)
        pose_config.TEST.MODEL_FILE = pose_model_path
        self.skeleton_model.load_state_dict(torch.load(pose_config.TEST.MODEL_FILE))
        self.scoremap_process = SkeletonScoremap(
            normalize_heatmap=True, group_mode='sum', gaussion_smooth=None)
        # should be always in eval mode
        self.skeleton_model = self.skeleton_model.eval()
        self.scoremap_process = self.scoremap_process.eval()

        # multiple heads, support BNHead only
        self.HEAD_NUM = 14 # includes 13 local features and 1 global features
        for idx in range(self.HEAD_NUM):
            multihead_cfg = {
                'name': 'BNHead',
                'in_dim': in_dim,
                'class_num': class_num,
                'classifier': classifier,
            }
            setattr(self, 'head{}'.format(idx), build_head(**multihead_cfg))

        self.disable_hash()


    def forward(self, imgs, feats_map, y=None):
        """
        Args:
            imgs(torch.tensor): [bs, cI, hI, wI]
            feats_map(torch.tensor): [bs, cF, hF, wF]
            y(torch.tensor): labels, [bs, num_classes]
        Return:
            res(dict):
                when training: {
                'feats_list': [[], [], ...],
                'bnfeats_list': [[], [], ...],
                'logits_list': [[], [], ...],
                ...
                }
                when testing: {
                'feats': [],
                'bn_feats': [],
                ...
                }
        """

        # should be always in eval mode
        self.skeleton_model = self.skeleton_model.eval()
        self.scoremap_process = self.scoremap_process.eval()

        # compute local features and global features
        with torch.no_grad():
            heat_maps = self.skeleton_model(imgs)
        feats_vec_list, confidences = \
            self._compute_skeleton_local_feats_vec(feats_map, heat_maps)

        # heads
        res_tmp = []
        for idx in range(self.HEAD_NUM):
            feats_vec = feats_vec_list[idx]
            head = getattr(self, 'head{}'.format(idx))
            r = head(feats_vec, y)
            res_tmp.append(r)
        keys = res_tmp[0].keys()

        # if test mode
        if not self.training:
            res = {}
            for key in keys:
                res[key] = self._normalize_and_weight_feature_vectors([r[key] for r in res_tmp], confidences, self.test_mode)
            return res

        # if training mode
        res = {}
        for key in keys:
            res[key+'_list'] = [r[key] for r in res_tmp]
        res['confidences'] = confidences

        return res

    def _compute_skeleton_local_feats_vec(self, feature_maps, heat_maps):

        # get score map
        with torch.no_grad():
            heat_maps = heat_maps.cuda(0)
            score_maps, score_confidence, key_points_location = self.scoremap_process(heat_maps)

        #  using detach to avoid backwords propagation to skeleton model
        score_maps = score_maps.detach()
        score_confidence = score_confidence.detach()
        key_points_location = key_points_location.detach()

        n, cs, hs, ws = score_maps.shape
        nt, ct, ht, wt = feature_maps.shape
        assert hs == ht and ws == wt, "resolution of skeleton heatmap and reid tensor should be the same"

        # get feature_vector_list
        feature_vector_list = []
        for i in range(self.HEAD_NUM):

            if i < score_maps.shape[1]:  # skeleton based local feature vectors
                score_map_i = score_maps[:, i, :, :].unsqueeze(1).repeat([1, 2048, 1, 1])
                feature_vector_i = torch.sum(score_map_i * feature_maps, [2, 3])
                feature_vector_list.append(feature_vector_i)

            else:  # global feature vectors
                feature_vector_i = torch.mean(feature_maps, [2, 3])
                feature_vector_i = feature_vector_i + F.adaptive_max_pool2d(feature_maps, 1).squeeze(-1).squeeze(-1)

                feature_vector_list.append(feature_vector_i)
                device = torch.device('cuda')
                score_confidence = torch.cat(
                    [score_confidence, torch.ones([score_confidence.shape[0], 1]).to(device)], dim=1)

        assert score_confidence.shape[1] == self.HEAD_NUM
        skeleton_num = score_maps.shape[1]
        score_confidence[:, skeleton_num:] = \
            F.normalize(score_confidence[:, skeleton_num:], 1, 1) * self.weight_global  # global feature score_confidence
        score_confidence[:, :skeleton_num] = \
            F.normalize(score_confidence[:, :skeleton_num], 1, 1)  # partial feature score_confidence
        return feature_vector_list, score_confidence

    def _normalize_and_weight_feature_vectors(self, feature_list, feature_confidence, test_feat_mode='sum-all'):
        skeleton_group_num = len(pose_config.MODEL.JOINTS_GROUPS)
        skeleton_feature_list = []
        global_feature_list = []
        for i, feature in enumerate(feature_list):
            weight = feature_confidence[:, i].view([feature_confidence.shape[0], 1]).repeat([1, feature.shape[1]])
            weighted_feature = torch.sqrt(weight) * F.normalize(feature, 1, 1)
            if i < skeleton_group_num:
                skeleton_feature_list.append(weighted_feature.unsqueeze(2))
            else:
                global_feature_list.append(weighted_feature.unsqueeze(2))

        skeleton_feature_vectors = torch.cat(skeleton_feature_list, dim=2)
        global_feature_vectors = torch.cat(global_feature_list, dim=2)

        if 'concat' in test_feat_mode:
            feature_vectors = torch.cat([skeleton_feature_vectors, global_feature_vectors], dim=2).reshape(
                feature_confidence.shape[0], -1)
        elif 'all' in test_feat_mode:
            feature_vectors = torch.cat([global_feature_vectors, skeleton_feature_vectors], dim=2)
            if 'max' in test_feat_mode:
                feature_vectors, _ = torch.max(feature_vectors, dim=2)
            elif 'sum' in test_feat_mode:
                feature_vectors = torch.sum(feature_vectors, dim=2)
        else:
            if 'sum' in test_feat_mode:
                skeleton_feature_vectors = torch.sum(skeleton_feature_vectors, dim=2)
                global_feature_vectors = torch.sum(global_feature_vectors, dim=2)
            elif 'max' in test_feat_mode:
                skeleton_feature_vectors, _ = torch.max(skeleton_feature_vectors, dim=2)
                global_feature_vectors, _ = torch.max(global_feature_vectors, dim=2)

            feature_vectors = torch.cat([skeleton_feature_vectors, global_feature_vectors], dim=1)
        return feature_vectors


    def enable_hash(self):
        for idx in range(self.HEAD_NUM):
            head = getattr(self, 'head{}'.format(idx))
            head.enable_hash()

    def disable_hash(self):
        for idx in range(self.HEAD_NUM):
            head = getattr(self, 'head{}'.format(idx))
            head.disable_hash()
