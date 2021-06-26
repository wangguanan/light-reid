# light-reid
a toolbox of light reid for fast feature extraction and search 
- [x] light-model: model distillation (3x faster feature extraction)
- [x] light-feature: binary code learning (6x faster retrieval)
- [x] light-search: [coarse2fine search](https://arxiv.org/abs/2008.06826) (2x faster retrieval)

it features
- [x] easy switch between light and non-light reid
- [x] simple modules for reid implementation
- [x] implementations of state-of-the-art deep reid models 


## What's New
- [2021.06]: we support [vision transformers](https://arxiv.org/abs/2010.11929) as cnn backbones. please refer [examples/bagtricks_buildwithconfigs/configs/base_config_duke_vit.yaml](./examples/bagtricks_buildwithconfigs/configs/base_config_duke_vit.yaml)
- [2020.12]: we release a strong pipeline for occluded/partial re-id. please refer [occluded_reid](./examples/occluded_reid)
- [2020.11]: we support pca_reduction to 128d with almost no accuracy drop. please refer [bagtricks_pca](./examples/bagtricks_pca)
- [2020.11]: we support build with config files, making coding more simple. please refer [bagtricks_buildwithconfigs](./examples/bagtricks_buildwithconfigs)
- [2020.08]: We release a toolbox of light-reid learning for faster inference, getting >30x faster speed.
- [2020.03]: We implement [BagTricks](https://ieeexplore.ieee.org/document/8930088) and support IBN-Net, MSMT17, combineall, multi-dataset train. Please see branch [version_py3.7_bot](https://github.com/wangguanan/light-reid/tree/version_py3.7_bot).
- [2019.03]: We give a clean implemention of  [BagTricks](https://ieeexplore.ieee.org/document/8930088) with python2.7. Please see branch [version_py2.7](https://github.com/wangguanan/light-reid/tree/version_py2.7).


## Find our Works
* [2020.07]: **[ECCV'20]** Our work about Fast ReID has been accepted by ECCV'20. ([Paper](https://arxiv.org/abs/2008.06826), [Code](https://github.com/wangguanan/light-reid))
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
pip install -r requirements

# install torch and torchvision (select the proper cuda version to suit your machine)
conda install pytorch==1.4.0 torchvision -c pytorch
# install faiss for stable search
conda install faiss-cpu -c pytorch
```

## Prepare Datasets

- download datasets that you need, [reid_datasets.md](./reid_datasets.md) lists various of datasets and their links.
- update datasets path at [./lightreid/data/datasets/datasetpaths.yaml](./lightreid/data/datasets/datasetpaths.yaml)

## Quick Start 
[1 step](./examples/bagtricks_buildwithconfigs) to build a SOTA reid model with configs


## Implemented reid methods and experimental results

- [x] [bagtricks_buildwithconfigs](./examples/bagtricks_buildwithconfigs): easily implement a strong reid baseline
- [x] [bagtricks_pca](./examples/bagtricks_pca): reduce feature dimension with PCA
- [x] [occluded_reid](./examples/occluded_reid): a simple&strong reid baseline for occluded reid
- [x] [generalizable_reid](./examples/generalizable_reid): a reid model performs well on multiple datasets

## Acknowledge

Our [light-reid](https://github.com/wangguanan/light-reid) partially refers open-sourced 
[torch-reid](https://github.com/KaiyangZhou/deep-person-reid) and 
[fast-reid](https://github.com/JDAI-CV/fast-reid),
we thank their awesome contribution to reid community. 

If you have any question about this reid toolbox, please feel free to contact me.
E-mail: guan.wang0706@gmail.com


## LICENSE
light-reid is released released under the MIT License (refer to the LICENSE file for details).


## Citation
if you find the repo is useful, please kindly cite our works
```
@article{wang2020faster,
 title="Faster Person Re-Identification.",
 author="Guan'an {Wang} and Shaogang {Gong} and Jian {Cheng} and Zengguang {Hou}",
 journal="In Proceedings of the European Conference on Computer Vision (ECCV)",
 year="2020"
}

@article{wang2020honet,
 title="High-Order Information Matters: Learning Relation and Topology for Occluded Person Re-Identification.",
 author="Guan'an {Wang} and Shuo {Yang} and Huanyu {Liu} and Zhicheng {Wang} and Yang {Yang} and Shuliang {Wang} and Gang {Yu} and Erjin {Zhou} and Jian {Sun}",
 journal="In Proceedings of the 2020 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
 year="2020"
}
```
