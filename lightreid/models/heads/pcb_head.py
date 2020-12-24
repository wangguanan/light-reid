import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from lightreid.utils import weights_init_kaiming, weights_init_classifier

from .build import HEADs_REGISTRY


@HEADs_REGISTRY.register()
class BottleClassifier(nn.Module):

    def __init__(self, in_dim, out_dim, relu=True, dropout=True, bottle_dim=512):
        super(BottleClassifier, self).__init__()

        bottle = [nn.Linear(in_dim, bottle_dim)]
        bottle += [nn.BatchNorm1d(bottle_dim)]
        if relu:
            bottle += [nn.LeakyReLU(0.1)]
        if dropout:
            bottle += [nn.Dropout(p=0.5)]
        bottle = nn.Sequential(*bottle)
        bottle.apply(weights_init_kaiming)
        self.bottle = bottle

        classifier = [nn.Linear(bottle_dim, out_dim)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.bottle(x)
        x = self.classifier(x)
        return x


class PCBHead(nn.Module):

    def __init__(self, in_dim, class_num, part_num):
        super(PCBHead, self).__init__()
        self.part_num = part_num
        # classifier
        for i in range(part_num):
            name = 'classifier' + str(i)
            setattr(self, name, BottleClassifier(in_dim, class_num, relu=True, dropout=False, bottle_dim=256))
        # embedder
        for i in range(part_num):
            name = 'embedder' + str(i)
            setattr(self, name, nn.Linear(in_dim, 256))

    def forward(self, feat, y=None):

        logits_list, embeddings_list = [], []
        for idx in range(self.part_num):
            feat_i = feat[:, :, idx]
            classifier_i = getattr(self, 'classifier'+str(idx))
            logits_i = classifier_i(feat_i)
            logits_list.append(logits_i)

            embedder_i = getattr(self, 'embedder'+str(idx))
            embedding_i = embedder_i(feat_i)
            embeddings_list.append(embedding_i)

        if not self.training:
            return F.normalize(feat, 2, dim=2).reshape([-1, feat.shape[1]*feat.shape[2]])
        return embeddings_list, logits_list
