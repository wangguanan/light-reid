# light-reid
a toolbox of light reid for fast feature extraction and search 
- [x] light-model: model distillation (2x-3x faster feature extraction)
- [x] light-feature: binary code learning (10x faster retrieval)
- [x] light-search: coarse2fine search (5x faster retrieval)

it features
- [x] easy switch between light and non-light reid
- [x] simple modules for reid implementation
- [x] implementations of state-of-the-art deep reid models 


## What's New
- [2020.08]: We release a toolbox of light-reid learning for faster inference, getting >50x faster speed.
- [2020.03]: We implement [BagTricks](https://ieeexplore.ieee.org/document/8930088) and support IBN-Net, MSMT17, combineall, multi-dataset train. Please see branch [version_py3.7_bot](https://github.com/wangguanan/light-reid/tree/version_py3.7_bot).
- [2019.03]: We give a clearn implemention of  [BagTricks](https://ieeexplore.ieee.org/document/8930088) with python2.7. Please see branch [version_py2.7](https://github.com/wangguanan/light-reid/tree/version_py2.7).

## Find our Works
* [2020.07]: **[ECCV'20]** Our new work about Fast ReID has been accepted by ECCV'20. ([Paper] comming soon)
* [2020.03]: **[CVPR'20]** Our new work about Occluded ReID has been accepted by CVPR'20. ([Paper](https://arxiv.org/abs/2003.08177), [Code](https://github.com/wangguanan/HOReID)).
* [2020.01]: **[AAAI'20]** Our new work about RGB-Infrared(IR) ReID has been accepted by AAAI'20. ([Paper](https://arxiv.org/pdf/2002.04114.pdf), [Code](https://github.com/wangguanan/JSIA-ReID)).
* [2019.10]: **[ICCV'19]** Our new work about RGB-Infrared(IR) ReID has been accepted by ICCV'19. ([Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_RGB-Infrared_Cross-Modality_Person_Re-Identification_via_Joint_Pixel_and_Feature_Alignment_ICCV_2019_paper.pdf), [Code](https://github.com/wangguanan/AlignGAN)).
* [2019.05]: We implement PCB and achieve better performance than the offical one. ([Code](https://github.com/wangguanan/Pytorch-Person-ReID-Baseline-PCB-Beyond-Part-Models))


## Set Up
```shell script
conda create -n lightreid python=3.7
conda activate lightreid
conda install pytorch==1.4.0 torchvision -c pytorch
pip install matplotlib scipy Pillow numpy prettytable easydict scikit-learn gdown hexhamming progressbar2 pip install pyyaml
```


## Quick Start 
5 steps to implement a SOTA reid model [[readme](./examples/bagtricks/main.py)]


## Experimental Results and Trained Models

### Market-1501

comming soon

### DukeMTMC-reID

comming soon


## Contact
If you have any question about the project, please feel free to contact me.

E-mail: guan.wang0706@gmail.com