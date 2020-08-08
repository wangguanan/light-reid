## train a vanilla model
CUDA_VISIBLE_DEVICES=0 python main.py --results_dir ./results/

# enable lightmodel, enhance with model distillation
CUDA_VISIBLE_DEVICES=0 python main.py --results_dir ./results/ --lightmodel True

## enable lightfeat, learn binary codes
CUDA_VISIBLE_DEVICES=1 python main.py --results_dir ./results/ --lightfeat True
#
## enable lightfeat and lightsearch, learn binary codes and search with coarse2fine
CUDA_VISIBLE_DEVICES=2 python main.py --results_dir ./results/ --lightfeat True --lightsearch True