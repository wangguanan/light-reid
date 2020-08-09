import os
import sys
sys.path.append('../..')
import torch
import torch.nn as nn
import lightreid


# build dataset
MARKET_PAH = '/raid/Monday/DataSets/Market-1501/Market-1501-v15.09.15/'
DUKE_PATH = '/raid/Monday/DataSets/duke/DukeMTMC-reID/'
datamanager = lightreid.data.DataManager(
    sources=[lightreid.data.Market1501(data_path=MARKET_PAH, combineall=False)],
    target=lightreid.data.Market1501(data_path=MARKET_PAH, combineall=False),
    transforms_train=lightreid.data.build_transforms(img_size=[384, 128], transforms_list=['autoaug', 'rea'], total_epochs=90),
    transforms_test=lightreid.data.build_transforms(img_size=[384, 128], transforms_list=[]),
    sampler='pk', p=4, k=16)

# build model
backbone = lightreid.models.backbones.resnet50(pretrained=True, last_stride_one=True)
pooling = lightreid.models.GeneralizedMeanPoolingP()
head = lightreid.models.BNNeckHead(backbone.dim, class_num=datamanager.class_num,
       classifier=lightreid.models.Circle(backbone.dim, datamanager.class_num, scale=64, margin=0.35))
model = lightreid.models.BaseReIDModel(backbone=backbone, pooling=pooling, head=head)

# build loss
criterion = lightreid.losses.Criterion([
    {'criterion': lightreid.losses.CrossEntropyLabelSmooth(num_classes=datamanager.class_num, epsilon=0.1), 'weight': 1.0},
    {'criterion': lightreid.losses.TripletLoss(margin='soft', metric='euclidean'), 'weight': 1.0},
])

# build optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.00035, weight_decay=5e-4)
lr_scheduler = lightreid.optim.DelayedCosineAnnealingLR(
    optimizer, delay_epochs=45, max_epochs=90, eta_min_lr=0.00000077,
    warmup_factor=0.001, warmup_epochs=10, warmup_method='linear')
optimizer = lightreid.optim.Optimizer(optimizer=optimizer, lr_scheduler=lr_scheduler, max_epochs=90, fix_cnn_epochs=10)

# run
solver = lightreid.engine.Engine(
    results_dir='./results2/', datamanager=datamanager, model=model, criterion=criterion, optimizer=optimizer, use_gpu=True)
solver.train()
solver.eval()
