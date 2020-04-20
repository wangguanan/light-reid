'''
this file provide operation of extracting a list of images
and return their features
'''

import torch
import numpy as np
import argparse
from core import build_extractor

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128], help='should be consistent with pre-trained model')
parser.add_argument('--pid_num', type=int, default=751, help='751 for Market-1501, 702(maybe 751) for DukeMTMC-reID')
parser.add_argument('--model_path', type=str, default='/data/projects/20200217_reid4tracking/3.0-reid/4.0-bot/results/market/model-market.pkl', help='pre-trained model path')
config = parser.parse_args()

ReIDExtractor = build_extractor(config, use_cuda=True)
images = [np.random.randint(0, 255, [3, 22, 54]) for _ in range(1)]
feature_list = ReIDExtractor.extract_list(images)
print(feature_list)