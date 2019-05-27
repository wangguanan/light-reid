## Pytorch-Person-ReID-Baseline-Bag-of-Tricks
* **Introduction**: This project re-implements the strong person re-identification baseline: Bag of Tricks ([paper](https://arxiv.org/abs/1903.07071) and [official code](https://github.com/michuanhaohao/reid-strong-baseline)).
* **Advantage**: This project provides a **more simple and clear implementation** by only using the best parameters and removing lots of unnecessary modules.
* **Acknowledge**: This project refers the [official code](https://github.com/michuanhaohao/reid-strong-baseline), if you find this project useful, please cite the offical paper.
    ```
    @inproceedings{luo2019bag,
      title={Bag of Tricks and A Strong Baseline for Deep Person Re-identification},
      author={Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
      year={2019}
    }
    ```

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

### Tricks we used
* [x] Warm up learning rate
* [x] Random erasing augmentation (REA)
* [x] Label smoothing
* [x] Last stride
* [x] BNNeck
* [x] **Note that** our implementation uses no the center loss and re-ranking.
### Settings
* We conduct our experiments on 1 GTX1080ti GPU

### Results (with REA)

| Repeat | market2market | market2duke | | duke2duke |  duke2market |
| :---:                             | :---: | :---: | - |:---: | :---: |
| 1 | 0.939 (0.858) | 0.290 (0.159) | | 0.874 (0.767) | 0.486 (0.210) | 
| 2 | 0.944 (0.858) | 0.295 (0.156) | | 0.868 (0.765) | 0.492 (0.223) |
| 3 | 0.942 (0.859) | 0.281 (0.152) | | 0.863 (0.765) | 0.485 (0.221) |
| Average | 0.942 (0.858) | 0.289 (0.156) | | 0.868 (0.766) | 0.488 (0.218) |
| Paper | 0.941 (0.857) | - | | 0.864 (0.764) |

### Results (without REA)
| Repeat | market2market | market2duke | | duke2duke |  duke2market |
| :---:                             | :---: | :---: | - |:---: | :---: |
| 1 | 0.936 (0.824) | 0.427 (0.264) | | 0.849 (0.714) | 0.556 (0.269) |
| Paper | - | 0.414(0.257) | | - | 0.543 (0.255) |  

## Contacts
If you have any question about the project, please feel free to contact with me.

E-mail: guan.wang0706@gmail.com
