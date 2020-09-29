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

# init dataset paths
with open(args.config_file) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
# solver = lightreid.build_engine(config)

# # train
# solver.train(eval_freq=10)
# # test
# solver.resume_latest_model()
# solver.smart_eval(onebyone=True, mode='normal')

# inference only
MODEL_PATH = '/data/projects/light-reid/light-reid/examples/test/results/market1501+msmt17+wildtrackcrop+rap(combineall)-colorjitor-removerea-njust365win-ibna-dataparallel-gpu2-p24k6-circle-60epochs-cuhk03-resnet50/lightmodel(False)-lightfeat(False)-lightsearch(False)/model_60.pth'
inference = lightreid.build_inference(config, model_path=MODEL_PATH, use_gpu=False)

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


import faiss
import numpy as np
import copy

# index
ts = time.time()
features = features[:, :512]
query_f = features
gallery_f = np.concatenate([features, np.random.randn(100000, features.shape[1])], axis=0).astype(np.float32)
index = faiss.IndexFlatL2(query_f.shape[1])
index.add(gallery_f)
print('construct gallery, spend {}s, gallery size {}'.format(time.time()-ts, gallery_f.shape))

#
ts = time.time()
k = 4 # we want to see 4 nearest neighbors
D, I = index.search(query_f, k) # sanity check
# print(I)
# print(D)
print('search gallery, spend {}s'.format(time.time()-ts))

