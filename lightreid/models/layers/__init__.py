from .circle import Circle
from .arcface import ArcFace
from .generalize_mean_pooling import GeneralizedMeanPoolingP, GeneralizedMeanPooling
from .identity_pooling import IdentityPooling

import torch
import torch.nn as nn

from .build_pooling import build_pooling
from .build_classifier import build_classifier
