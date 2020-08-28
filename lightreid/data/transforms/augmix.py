# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base augmentations operators."""

import numpy as np
from PIL import Image
from PIL import ImageOps
import torch
import random


# ImageNet code should change this value
IMAGE_SIZE = [256, 128]


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  ret = ImageOps.posterize(pil_img, 4 - level)
  return ret

def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  ret = pil_img.rotate(degrees, resample=Image.BILINEAR)
  return ret

def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  ret = ImageOps.solarize(pil_img, 256 - level)
  return ret

def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  ret  = pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)
  return ret

def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  ret = pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)
  return ret

def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), pil_img.size[1] / 3)
  if np.random.random() > 0.5:
    level = -level
  ret = pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)
  return ret

def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), pil_img.size[1] / 3)
  if np.random.random() > 0.5:
    level = -level
  ret = pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)
  return ret
augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

class AugMix(object):
    #   Args:
    #     image: PIL.Image input image
    #     preprocess: Preprocessing function which should return a torch tensor.
    #
    #   Returns:
    #     mixed: Augmented and mixed image.
    #

    def __init__(self, prob=0.5, aug_prob_coeff = 0.1,
                        mixture_width = 3,
                        mixture_depth = 1,
                        aug_severity = 1):
        self.prob = prob
        self.aug_prob_coeff = aug_prob_coeff
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_severity = aug_severity

    def __call__(self, img):
        if random.random() > self.prob:
            return np.asarray(img)
        ws = np.float32(
            np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

        mix = np.zeros([img.size[1], img.size[0], 3])
        for i in range(self.mixture_width):
            image_aug = img.copy()
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(
                1, 4)
            for _ in range(depth):
                op = np.random.choice(augmentations)
                image_aug = op(image_aug, self.aug_severity)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * np.asarray(image_aug)

        mixed = (1 - m) * np.asarray(img) + m * mix
        return mixed.astype(np.uint8)