## Pytorch-Person-REID-Baseline-Bag-of-Tricks
* This project re-implements the strong person re-identification baseline [Bag of Tricks (BoT)](https://arxiv.org/abs/1903.07071), and refers the [offical code](https://github.com/michuanhaohao/reid-strong-baseline) a lot.
* **Advantage**: Compared with the offical implementation, his project provides a more simple and clear implementation of BoT by removing unnecessary modules.

## Dependencies
* [Anaconda (Python 2.7)](https://www.anaconda.com/download/)
* [PyTorch 0.4.0](http://pytorch.org/)
* GPU Memory >= 10G
* Memory >= 20G

## Dataset Preparation
* [Market-1501 Dataset](http://ww7.liangzheng.org/) and [DukeMTMC-reID Dataset](https://github.com/layumi/DukeMTMC-reID_evaluation)
* Download and extract both anywhere

## Train and Test
```
python main.py --market_path market_path --duke_path duke_path
```

## Experiments

### Settings
* We conduct our experiments on 1 GTX1080ti GPUs

### Results

| Repeat | market2market | duke2duke | market2duke | duke2market |
| ---                               | :---: | :---: | :---: | :---: |
| 1 | 0.939 (0.858) | 0.874 (0.767) | 0.290 (0.159) | 0.486 (0.210) | 
| 2 | 0.944 (0.858) | 0.868 (0.765) | 0.295 (0.156) | 0.492 (0.223) |
| 3 | 0.942 (0.859) | 0.863 (0.765) | 0.281 (0.152) | 0.485 (0.221) |

## Contacts
If you have any question about the project, please feel free to contact with me.

E-mail: guan.wang0706@gmail.com
