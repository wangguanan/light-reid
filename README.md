# light-reid
a toolbox of light reid for fast feature extraction and search 
- [x] light-model: model distillation (3x faster feature extraction)
- [x] light-feature: binary code learning (6x faster retrieval)
- [x] light-search: coarse2fine search (2x faster retrieval)

it features
- [x] easy switch between light and non-light reid
- [x] simple modules for reid implementation
- [x] implementations of state-of-the-art deep reid models 

references
- [ECCV'20] Faster Person Re-Identification. Guan'an Wang, Shaogang Gong, Jian Cheng and Zengguang Hou. [[paper]](https://arxiv.org/abs/2008.06826)

## What's New
- [2020.08]: We release a toolbox of light-reid learning for faster inference, getting >30x faster speed.
- [2020.03]: We implement [BagTricks](https://ieeexplore.ieee.org/document/8930088) and support IBN-Net, MSMT17, combineall, multi-dataset train. Please see branch [version_py3.7_bot](https://github.com/wangguanan/light-reid/tree/version_py3.7_bot).
- [2019.03]: We give a clearn implemention of  [BagTricks](https://ieeexplore.ieee.org/document/8930088) with python2.7. Please see branch [version_py2.7](https://github.com/wangguanan/light-reid/tree/version_py2.7).


## Find our Works
* [2020.07]: **[ECCV'20]** Our work about Fast ReID has been accepted by ECCV'20. ([Paper](https://arxiv.org/abs/2008.06826) [Code](https://github.com/wangguanan/light-reid))
* [2020.03]: **[CVPR'20]** Our work about Occluded ReID has been accepted by CVPR'20. ([Paper](https://arxiv.org/abs/2003.08177), [Code](https://github.com/wangguanan/HOReID)).
* [2020.01]: **[AAAI'20]** Our work about RGB-Infrared(IR) ReID has been accepted by AAAI'20. ([Paper](https://arxiv.org/pdf/2002.04114.pdf), [Code](https://github.com/wangguanan/JSIA-ReID)).
* [2019.10]: **[ICCV'19]** Our work about RGB-Infrared(IR) ReID has been accepted by ICCV'19. ([Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_RGB-Infrared_Cross-Modality_Person_Re-Identification_via_Joint_Pixel_and_Feature_Alignment_ICCV_2019_paper.pdf), [Code](https://github.com/wangguanan/AlignGAN)).
* [2019.05]: We implement PCB and achieve better performance than the offical one. ([Code](https://github.com/wangguanan/Pytorch-Person-ReID-Baseline-PCB-Beyond-Part-Models))


## Installation
```shell script
# clone this repo
git clone https://github.com/wangguanan/light-reid.git

# create environment
cd light-reid
conda create -n lightreid python=3.7
conda activate lightreid

# install dependencies
pip install -r requirements.txt

# install torch and torchvision (select the proper cuda version to suit your machine)
conda install pytorch==1.4.0 torchvision -c pytorch
```


## Quick Start 
[5 steps](./examples/bagtricks/main.py) to implement a SOTA reid model 


## Implemented reid methods and experimental results

- [x] [bagtricks](./examples/bagtricks)
- [ ] [agw](./examples/agw)
- [ ] sbs (constructing)
- [ ] pcb (todo)
- [ ] mgn (todo)

## Acknowledge

Our [light-reid](https://github.com/wangguanan/light-reid) partially refers open-sourced 
[torch-reid](https://github.com/KaiyangZhou/deep-person-reid) and 
[fast-reid](https://github.com/JDAI-CV/fast-reid),
we thank their awesome contribution to reid community. 

If you have any question about this reid toolbox, please feel free to contact me.
E-mail: guan.wang0706@gmail.com