## Pytorch-Person-ReID-Baseline-Bag-of-Tricks
* **Introduction**: This project re-implements the strong person re-identification baseline: Bag of Tricks ([paper](https://arxiv.org/abs/1903.07071) and [official code](https://github.com/michuanhaohao/reid-strong-baseline)).
* **Advantage**: This project provides a **more simple and clear implementation** by only using the best parameters and removing lots of unnecessary modules.
[](2.1-bot)

## Properties
* **Dataset**: Support multiple Datasets
  - [x] Market-1501, DukeMTMC-reID and MSMT17
  - [x] [Optional] Combine train/query/gallery as training set, i.e. ```train = train + query + gallery```
  - [x] [Optional] Comine multiple datasets as a big dataset, e.g. ```dataset = market + duke + msmt```
* **Models**: Support multiple CNN Backbones
  - [x] ResNet50
  - [x] ResNet50-ibna
  - [x] OSNet-AIN
* **Evaluation**: Support multiple evaluation protocols 
  - [x] MAP and CMC
  - [x] Precision and Recall
  - [x] Inter-Camera, Intra-Camera, ALL
  - [x] Visualize Ranking List

## News
* 2020-03-27: **[CVPR'20]** Our new work about Occluded ReID has been accepted by CVPR'20. ([Paper](https://arxiv.org/abs/2003.08177))
* 2020-01-01: **[AAAI'20]** Our new work about RGB-Infrared(IR) ReID for dark situation has been accepted by AAAI'20. ([Paper](https://arxiv.org/pdf/2002.04114.pdf), [Code](https://github.com/wangguanan/JSIA-ReID)).
* 2019-10-25: **[ICCV'19]** Our new work about RGB-Infrared(IR) ReID for dark situation has been accepted by ICCV'19. ([Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_RGB-Infrared_Cross-Modality_Person_Re-Identification_via_Joint_Pixel_and_Feature_Alignment_ICCV_2019_paper.pdf), [Code](https://github.com/wangguanan/AlignGAN)).
* 2019-05-01: We implement PCB and achieve better performance than the offical one. ([Code](https://github.com/wangguanan/Pytorch-Person-ReID-Baseline-PCB-Beyond-Part-Models))

## Update
* 2020-05-15: support [IBN-Net](https://github.com/XingangPan/IBN-Net) as cnnbackbone, support MSMT17 dataset, support train+query+gallery as trainset, support multi-dataset train
* 2020-03-27: we change the dependency to _Python3.7_ and _PyTorch-1.1.0._ If you want the old version depending on _Python-2.7_ and _PyTorch-0.4.0_, please find on [verion_py27 branch](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/tree/version_py2.7).
* 2019-06-18: we add visualization code to show ranked images 

## Dependencies
* [Anaconda (Python 3.7)](https://www.anaconda.com/download/)
* [PyTorch 1.1.0](http://pytorch.org/)
* PrettyTable (```pip install prettytable```)
* GPU Memory >= 10G
* Memory >= 10G

## Dataset Preparation
* Market-1501 ([Project](http://www.liangzheng.com.cn/Project/project_reid.html), [Google Drive](https://drive.google.com/open?id=1M8m1SYjx15Yi12-XJ-TV6nVJ_ID1dNN5))
* DukeMTMC-reID ([Project](https://github.com/sxzrt/DukeMTMC-reID_evaluation), [Google Drive](https://drive.google.com/open?id=11FxmKe6SZ55DSeKigEtkb-xQwuq6hOkE))
* MSMT17 ([Project](https://www.pkuvmc.com/dataset.html), [Paper](https://arxiv.org/pdf/1711.08565.pdf), Google Drive \<please e-mail me for the link\>)
* Download and extract both anywhere

## Trained Models
* Trained model on Market-1501 [[link]](https://drive.google.com/open?id=1UEginjwTwNDonO9Sl9DD9R0bYpiSRFu3)
* Trained model on DukeMTMC-reID (comming soon)
* Trained model on MSMT17 (comming soon)

## Run
#### Train on Market-1501/DukeMTMC-reID/MTMC17
```
python3 main.py --mode train \
    --train_dataset market --test_dataset market \
    --market_path /path/to/market/dataset/ \
    --output_path ./results/market/ 
python3 main.py --mode train \
    --train_dataset duke --test_dataset duke \
    --duke_path /path/to/duke/dataset/ \
    --output_path ./results/duke/
python3 main.py --mode train \
    --train_dataset msmt --test_dataset msmt --steps 400 --pid_num 1041 \
    --duke_path /path/to/msmt/dataset/ \
    --output_path ./results/msmt/
```

#### Train with ResNet50-IBNa backbone
```
# download model to ./core/nets/models/ from https://drive.google.com/file/d/1_r4wp14hEMkABVow58Xr4mPg7gvgOMto/view
python3 main.py --mode train -cnnbackbone res50ibna \
    --train_dataset market --test_dataset market \
    --market_path /path/to/market/dataset/ \
    --output_path ./results/market/ 
```

#### Train 

#### Test on Market-1501/DukeMTMC-reID/MTMC-17
```
python3 main.py --mode test \
    --train_dataset market --test_dataset market \
    --market_path /path/to/market/dataset/ \
    --resume_test_model /path/to/trained/model.pkl \ 
    --output_path ./results/test-on-market/
python3 main.py --mode test \
    --train_dataset duke --test_dataset duke \
    --market_path /path/to/duke/dataset/ \
    --resume_test_model /path/to/trained/model.pkl \ 
    --output_path ./results/test-on-duke/
python3 main.py --mode test \
    --train_dataset msmt --test_dataset msmt --pid_num 1041 \
    --market_path /path/to/msmt/dataset/ \
    --resume_test_model /path/to/trained/model.pkl \ 
    --output_path ./results/test-on-msmt/
```

#### Visualize Market-1501/DukeMTMC-reID
```
python3 main.py --mode visualize --visualize_mode inter-camera \
    --train_dataset market --visualize_dataset market \
    --market_path /path/to/market/dataset/ \
    --resume_visualize_model /path/to/trained/model.pkl \ 
    --visualize_output_path ./results/vis-on-market/ 
python3 main.py --mode visualize --visualize_mode inter-camera \
    --train_dataset duke --visualize_dataset duke \
    --market_path /path/to/duke/dataset/ \
    --resume_visualize_model /path/to/trained/model.pkl \ 
    --visualize_output_path ./results/vis-on-duke/ 
```

#### Visualize Customed Dataset with Trained Model

```
# customed dataset structure
|____ data_path/
     |____ person_id_1/
          |____ pid_1_imgid_1.jpg
          |____ pid_1_imgid_2.jpg
          |____ ......
     |____ person_id_2/
     |____ person_id_3/
     |____ ......
```
```
python3 demo.py \
    --resume_visualize_model /path/to/pretrained/model.pkl \
    --query_path /path/to/query/dataset/ --gallery_path /path/to/gallery/dataset/ \
    --visualize_output_path ./results/vis-on-cus/
```

## Experiments

### 1. Tricks we used
* [x] Warm up learning rate
* [x] Random erasing augmentation (REA)
* [x] Label smoothing
* [x] Last stride
* [x] BNNeck
* [x] ColorJitor

### 2. Settings
* We conduct our experiments on 1 GTX1080ti GPU

### 3. Experimental Results
#### Results (with REA)

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

### Results (different CNN Backbone)
| cnn backbone | train       | cball | rea  | color | msmt            | market         | duke           | wt(inter)  | wt(all)    | model ([BaiduPan](https://pan.baidu.com/s/1s1k_Gfzj9TvCIXJlqyJnEg), vskp)                                |
| ------------ | ----------- | ----- | :--- | ----- | --------------- | -------------- | -------------- | ---------- | ---------- | ------------------------------------ |
| res50ibna    | market      | √     | x    | √     | 28.6(10.6)      | ~~99.8(99.7)~~ | 58.1(38.1)     | 20.1(10.6) | 80.7(23.1) | res50ibna_market_cball_color.pth     |
| res50ibna    | duke        | √     | x    | √     | -               | 67.0(36.4)     | ~~99.3(98.6)~~ | 35.2(15.3) | 87.0(30.0) | res50ibna_duke_cball_color.pth       |
| res50ibna    | msmt        | √     | x    | √     | ~~96.5(91.4)~~  | 75.3(48.1)     | 71.9(54.7)     | 29.1(23.4) | 79.6(36.2) | res50ibna_msmt_cball_color.pth       |
| res50ibna    | msmt+duke   | √     | x    | √     | ~~95.1(87.7)~~  | 77.7(51.3)     | ~~96.8(93.2)~~ | 35.4(28.7) | 82.6(40.6) | res50ibna_msmtduke_cball_color.pth   |
| res50ibna    | msmt+market | √     | x    | √     | ~~95.0(87.90)~~ | ~~98.7(96.5)~~ | 73.2(57.0)     | 31.2(26.8) | 81.1(38.8) | res50ibna_msmtmarket_cball_color.pth |
| osnetain     | msmt        | √     | x    | √     | ~~93.3(81.8)~~  | 71.2(43.4)     | 69.8(51.2)     | 27.2(22.0) | -          | osnetain_msmt_cball_color.pth        |
| osnetain     | msmt+duke   | √     | x    | √     | 87.1(67.4)      | ~~96.2(88.7)~~ | ~~88.7(78.8)~~ | 39.5(26.8) | -          | osnetain_msmtduke_cball_color.pth    |
| osnetain     | msmt+market | √     | x    | √     | 93.0(80.4)      | ~~98.7(94.5)~~ | 72.5(54.7)     | 31.1(23.8) | -          | osnetain_msmtmarket_cball_color.pth  |

### 4. Visualization of Ranked Images on Market-1501 Dataset (with REA)
![](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/ranked_images/market/1351_c6s3_064142_00.jpg?raw=true)
![](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/ranked_images/market/1354_c5s3_040965_00.jpg?raw=true)
![](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/ranked_images/market/1357_c5s3_094087_00.jpg?raw=true)
* More results can be seen in folder ```ranked_images/market```


## Contacts
If you have any question about the project, please feel free to contact with me.

E-mail: guan.wang0706@gmail.com