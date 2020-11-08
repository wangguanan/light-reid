import argparse, ast
import sys
sys.path.append('../..')
sys.path.append('.')
import torch
import lightreid
import yaml
import time
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='path/to/config.yaml', help='')
parser.add_argument('--model_path', type=str, default='path/to/model.pth', help='')
args = parser.parse_args()

# load configs from yaml
with open(args.config_file) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# inference only
inference = lightreid.build_inference(config, model_path=args.model_path, use_gpu=False)

# process
img_paths = [
    './imgs/3006_c1s1_f000.jpg',
    './imgs/3006_c2s1_f000.jpg',
    './imgs/3007_c1s1_f000.jpg',
    './imgs/3007_c2s1_f000.jpg',
    './imgs/3008_c1s1_f000.jpg',
    './imgs/3008_c2s1_f000.jpg',
    './imgs/3013_c1s1_f000.jpg',
    './imgs/3013_c2s1_f000.jpg',
]
features = inference.process(img_paths, return_type='numpy')

# compute distance
print('feature shape: {}'.format(features.shape))
cosine_similarity = np.matmul(features, features.transpose([1,0])) # inner-produce distance (i.e. cosine distance since the feature has been normalized)
print('cosine similarity as below: ')
print(cosine_similarity)
