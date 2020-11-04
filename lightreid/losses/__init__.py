"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

from .label_smooth_cross_entropy_loss import CrossEntropyLabelSmooth
from .triplet_loss_with_batchhard import TripletLoss
from .focal_loss import FocalLoss
from .center_loss import CenterLoss
from .self_distill_loss import ProbSelfDistillLoss, SIMSelfDistillLoss
from .kl_loss import KLLoss
from .criterion import Criterion, build_criterion

