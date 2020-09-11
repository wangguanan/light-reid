"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Optional


class Circle(nn.Module):
    def __init__(self, in_features, out_features, scale, margin):
        super().__init__()
        self.in_feat = in_features
        self._num_classes = out_features
        self._s = scale
        self._m = margin

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, features, targets):
        sim_mat = F.linear(F.normalize(features), F.normalize(self.weight))
        alpha_p = F.relu(-sim_mat.detach() + 1 + self._m)
        alpha_n = F.relu(sim_mat.detach() + self._m)
        delta_p = 1 - self._m
        delta_n = self._m

        s_p = self._s * alpha_p * (sim_mat - delta_p)
        s_n = self._s * alpha_n * (sim_mat - delta_n)

        targets = one_hot(targets, num_classes=self._num_classes)

        pred_class_logits = targets * s_p + (1.0 - targets) * s_n
        return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )


# based on:
# https://github.com/kornia/kornia/blob/master/kornia/utils/one_hot.py
def one_hot(labels: torch.Tensor,
            num_classes: int,
            dtype: Optional[torch.dtype] = None, ) -> torch.Tensor:
    # eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.
    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}".format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    device = labels.device
    shape = labels.shape
    one_hot = torch.zeros(shape[0], num_classes, *shape[1:],
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
