## Pytorch-Person-ReID-Baseline-Bag-of-Tricks
* **Introduction**: This project re-implements the strong person re-identification baseline: Bag of Tricks ([paper](https://arxiv.org/abs/1903.07071) and [official code](https://github.com/michuanhaohao/reid-strong-baseline)).
* **Advantage**: This project provides a **more simple and clear implementation** by only using the best parameters and removing lots of unnecessary modules.
[](2.1-bot)

## News
* 2020-03-27: **[CVPR'20]** Our new work about Occluded ReID has been accepted by CVPR'20. ([Paper](https://arxiv.org/abs/2003.08177))
* 2020-01-01: **[AAAI'20]** Our new work about RGB-Infrared(IR) ReID for dark situation has been accepted by AAAI'20. ([Paper](https://arxiv.org/pdf/2002.04114.pdf), [Code](https://github.com/wangguanan/JSIA-ReID)).
* 2019-10-25: **[ICCV'19]** Our new work about RGB-Infrared(IR) ReID for dark situation has been accepted by ICCV'19. ([Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_RGB-Infrared_Cross-Modality_Person_Re-Identification_via_Joint_Pixel_and_Feature_Alignment_ICCV_2019_paper.pdf), [Code](https://github.com/wangguanan/AlignGAN)).
* 2019-05-01: We implement PCB and achieve better performance than the offical one. ([Code](https://github.com/wangguanan/Pytorch-Person-ReID-Baseline-PCB-Beyond-Part-Models))

## Update
* 2020-05-15: support msmt dataset (optional combine all) and cnn backbone [IBN-Net](https://github.com/XingangPan/IBN-Net)
* 2020-03-27: add demo.py to visualize customed dataset.
* 2020-03-27: we change the dependency to _Python3.7_ and _PyTorch-1.1.0._ If you want the old version depending on _Python-2.7_ and _PyTorch-0.4.0_, please find on [verion_py27 branch](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/tree/version_py2.7).
* 2019-06-18: we add visualization code to show ranked images 

## Dependencies
* [Anaconda (Python 3.7)](https://www.anaconda.com/download/)
* [PyTorch 1.1.0](http://pytorch.org/)
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
python3 main.py --mode train -cnnbackbone res50ibna \
    --train_dataset market --test_dataset market \
    --market_path /path/to/market/dataset/ \
    --output_path ./results/market/ 
```

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

### 3. Results (with REA)

| Repeat | market2market | market2duke | | duke2duke |  duke2market |
| :---:                             | :---: | :---: | - |:---: | :---: |
| 1 | 0.939 (0.858) | 0.290 (0.159) | | 0.874 (0.767) | 0.486 (0.210) | 
| 2 | 0.944 (0.858) | 0.295 (0.156) | | 0.868 (0.765) | 0.492 (0.223) |
| 3 | 0.942 (0.859) | 0.281 (0.152) | | 0.863 (0.765) | 0.485 (0.221) |
| Average | 0.942 (0.858) | 0.289 (0.156) | | 0.868 (0.766) | 0.488 (0.218) |
| Paper | 0.941 (0.857) | - | | 0.864 (0.764) |

### 4. Results (without REA)
| Repeat | market2market | market2duke | | duke2duke |  duke2market |
| :---:                             | :---: | :---: | - |:---: | :---: |
| 1 | 0.936 (0.824) | 0.427 (0.264) | | 0.849 (0.714) | 0.556 (0.269) |
| Paper | - | 0.414(0.257) | | - | 0.543 (0.255) |  

### 5. Visualization of Ranked Images on Market-1501 Dataset (with REA)
![](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/ranked_images/market/1351_c6s3_064142_00.jpg?raw=true)
![](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/ranked_images/market/1354_c5s3_040965_00.jpg?raw=true)
![](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/ranked_images/market/1357_c5s3_094087_00.jpg?raw=true)
* More results can be seen in folder ```ranked_images/market```


## Contacts
If you have any question about the project, please feel free to contact with me.

E-mail: guan.wang0706@gmail.com
