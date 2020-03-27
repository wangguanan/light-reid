## Pytorch-Person-ReID-Baseline-Bag-of-Tricks
* **Introduction**: This project re-implements the strong person re-identification baseline: Bag of Tricks ([paper](https://arxiv.org/abs/1903.07071) and [official code](https://github.com/michuanhaohao/reid-strong-baseline)).
* **Advantage**: This project provides a **more simple and clear implementation** by only using the best parameters and removing lots of unnecessary modules.

## News
* 2020-03-27: add demo.py to visualize customed dataset.
* 2019-10-25: Our new work about RGB-Infrared(IR) ReID for dark situation has been accepted by ICCV'19. Code is available at  [here](https://github.com/wangguanan/AlignGAN).
* 2019-06-18: we add visualization code to show ranked images 
* 2019-05-01: We re-implement PCB and achieve better performance than the offical one. Code is available [here](https://github.com/wangguanan/Pytorch-Person-ReID-Baseline-PCB-Beyond-Part-Models).

## Dependencies
* [Anaconda (Python 3.7)](https://www.anaconda.com/download/)
* [PyTorch 1.1.0](http://pytorch.org/)
* GPU Memory >= 10G
* Memory >= 10G

## Dataset Preparation
* [Market-1501 Dataset](https://jingdongwang2017.github.io/Projects/ReID/Datasets/Market-1501.html) and [DukeMTMC-reID Dataset](https://github.com/layumi/DukeMTMC-reID_evaluation)
* Download and extract both anywhere

## Run
#### Train on Market-1501/DukeMTMC-reID
```
python3 main.py --mode train \
    --train_dataset market --test_dataset market \
    --market_path /path/to/market/ \
    --output_path ./results/market/ 
python3 main.py --mode train \
    --train_dataset duke --test_dataset duke \
    --duke_path /path/to/duke/ \
    --output_path ./results/duke/
```

#### Test on Market-1501/DukeMTMC-reID
```
python3 main.py --mode test \
    --train_dataset market --test_dataset market \
    --resume_test_model /path/to/pretrained/model.pkl \ 
    --output_path ./results/test-on-market/
python3 main.py --mode test\
    --train_dataset duke --test_dataset duke \
    --resume_test_model /path/to/pretrained/model.pkl \ 
    --output_path ./results/test-on-duke/
```

#### Visualize Market-1501/DukeMTMC-reID
```
python3 main.py --mode visualize --visualize_mode inter-camera \
    --train_dataset market --visualize_dataset market \
    --resume_visualize_model /path/to/pretrained/model.pkl \ 
    --visualize_output_path ./results/vis-on-market/ 
python3 main.py --mode visualize --visualize_mode inter-camera \
    --train_dataset duke --visualize_dataset duke \
    --resume_visualize_model /path/to/pretrained/model.pkl \ 
    --visualize_output_path ./results/vis-on-duke/ 
```

#### Visualize customed dataset

```
# customed dataset structure
|____ data_path/
     |____ person_id_1/
          |____ 1.jpg
          |____ 2.jpg
          |____ ......
     |____ person_id_2/
     |____ person_id_2/
     |____ ......
           
```
```
python3 demo.py \
    --resume_visualize_model /path/to/pretrained/model.pkl \
    --query_path /path/to/query/dataset/ --gallery_path /path/to/gallery/dataset/ 
    --visualize_output_path ./results/vis-on-cus/
```

## Experiments

### 1. Tricks we used
* [x] Warm up learning rate
* [x] Random erasing augmentation (REA)
* [x] Label smoothing
* [x] Last stride
* [x] BNNeck
* [x] **Note that** our implementation uses neither the center loss nor re-ranking.

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
| Query | Top1  | Top2  | Top3  | Top4  | Top5  | Top6  | Top7  | Top8  | Top9  | Top10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ![](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/ranked_images/market/0005_c1s1_001351_00.jpg/query_top000_name_0005_c1s1_001351_00.jpg?raw=true) | ![](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/ranked_images/market/0005_c1s1_001351_00.jpg/gallery_top001_name_0005_c3s2_088328_02.jpg?raw=true) | ![](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/ranked_images/market/0005_c1s1_001351_00.jpg/gallery_top002_name_0005_c5s1_000401_03.jpg?raw=true) | ![](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/ranked_images/market/0005_c1s1_001351_00.jpg/gallery_top003_name_0005_c5s1_000426_02.jpg?raw=true) | ![](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/ranked_images/market/0005_c1s1_001351_00.jpg/gallery_top004_name_0005_c4s1_006951_03.jpg?raw=true) | ![](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/ranked_images/market/0005_c1s1_001351_00.jpg/gallery_top005_name_0005_c5s1_001026_01.jpg?raw=true) | ![](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/ranked_images/market/0005_c1s1_001351_00.jpg/gallery_top006_name_0005_c3s3_060878_01.jpg?raw=true) | ![](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/ranked_images/market/0005_c1s1_001351_00.jpg/gallery_top007_name_0005_c5s1_001426_01.jpg?raw=true) | ![](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/ranked_images/market/0005_c1s1_001351_00.jpg/gallery_top008_name_0005_c5s1_014501_01.jpg?raw=true) | ![](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/ranked_images/market/0005_c1s1_001351_00.jpg/gallery_top009_name_0005_c5s1_000626_02.jpg?raw=true) | ![](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/ranked_images/market/0005_c1s1_001351_00.jpg/gallery_top010_name_0005_c3s3_074344_01.jpg?raw=true)

* More results can be seen in folder ```ranked_images/market```


## Contacts
If you have any question about the project, please feel free to contact with me.

E-mail: guan.wang0706@gmail.com
