import torch
import torch.nn as nn
import torchvision
from .bnneck import BNClassifier

class Res50BNNeck(nn.Module):

    def __init__(self, class_num):
        super(Res50BNNeck, self).__init__()

        self.class_num = class_num
        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        # cnn backbone
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool, # no relu
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # classifier
        self.classifier = BNClassifier(2048, self.class_num)

    def forward(self, x):

        features = self.gap(self.resnet_conv(x)).squeeze(dim=2).squeeze(dim=2)
        bned_features, cls_score = self.classifier(features)

        if self.training:
            return features, cls_score
        else:
            return bned_features