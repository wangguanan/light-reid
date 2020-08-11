### train a vanilla model
#CUDA_VISIBLE_DEVICES=1 python main.py --results_dir ./results/

## enable lightmodel, enhance with model distillation
CUDA_VISIBLE_DEVICES=1 python main.py --results_dir ./results/ --lightmodel True --circle_scale 64 --circle_margin 0.35

## enable lightmodel, enhance with model distillation
CUDA_VISIBLE_DEVICES=1 python main.py --results_dir ./results/ --lightmodel True --circle_scale 64 --circle_margin 0.25

## enable lightmodel, enhance with model distillation
CUDA_VISIBLE_DEVICES=1 python main.py --results_dir ./results/ --lightmodel True --circle_scale 64 --circle_margin 0.15

## enable lightmodel, enhance with model distillation
CUDA_VISIBLE_DEVICES=1 python main.py --results_dir ./results/ --lightmodel True --circle_scale 64 --circle_margin 0.05

## enable lightmodel, enhance with model distillation
CUDA_VISIBLE_DEVICES=1 python main.py --results_dir ./results/ --lightmodel True --circle_scale 32 --circle_margin 0.35

## enable lightmodel, enhance with model distillation
CUDA_VISIBLE_DEVICES=1 python main.py --results_dir ./results/ --lightmodel True --circle_scale 16 --circle_margin 0.35

## enable lightmodel, enhance with model distillation
CUDA_VISIBLE_DEVICES=1 python main.py --results_dir ./results/ --lightmodel True --circle_scale 8 --circle_margin 0.35

## enable lightmodel, enhance with model distillation
CUDA_VISIBLE_DEVICES=1 python main.py --results_dir ./results/ --lightmodel True --circle_scale 128 --circle_margin 0.35

#### enable lightfeat, learn binary codes
#CUDA_VISIBLE_DEVICES=1 python main.py --results_dir ./results/ --lightfeat True
##
#### enable lightfeat and lightsearch, learn binary codes and search with coarse2fine
#CUDA_VISIBLE_DEVICES=1 python main.py --results_dir ./results/ --lightfeat True --lightsearch True
#
### enable lightmodel, lightfeat and lightsearch, learn resnet18, binary codes and search with coarse2fine
CUDA_VISIBLE_DEVICES=1 python main.py --results_dir ./results/ --lightmodel True --lightfeat True --lightsearch True