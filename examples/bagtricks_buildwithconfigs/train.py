import argparse, ast
import sys
sys.path.append('../..')
sys.path.append('.')
import torch
import lightreid
import yaml
import time


parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='./configs/base_config.yaml', help='')
args = parser.parse_args()

# load configs from yaml file
with open(args.config_file) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
# init solver
solver = lightreid.build_engine(config)

# train
solver.train(eval_freq=10)

# test
solver.resume_latest_model()
solver.smart_eval(onebyone=True, mode='normal')

# visualize
solver.resume_latest_model()
solver.visualize()

