import torch
import torch.nn as nn


class IdentityPooling(nn.Module):
    """
    identity mapping, i.e. return input, do nothing
    """

    def __init__(self):
        super(IdentityPooling, self).__init__()

    def forward(self, x):
        return x