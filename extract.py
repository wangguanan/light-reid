'''
this file provide operation of extracting a list of images
and return their features
'''

import torch
import numpy as np
import argparse
from core import build_extractor

parser = argparse.ArgumentParser()
parser.add_argument('--cnnbackbone', type=str, default='res50', help='res50, res50ibna')
parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128], help='should be consistent with pre-trained model')
parser.add_argument('--pid_num', type=int, default=751, help='751 for Market-1501, 702(maybe 751) for DukeMTMC-reID')
parser.add_argument('--model_path', type=str, default='path/to/model.pkl', help='trained model path')
config = parser.parse_args()

ReIDExtractor = build_extractor(config, use_cuda=True)
images = [np.random.randint(0, 255, [128,128,3]) for _ in range(1)]
feature_list = ReIDExtractor.extract_list(images)
print(feature_list)