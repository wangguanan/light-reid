# demo for reid model

## update
- support index with [faiss](https://github.com/facebookresearch/faiss)（faster, more stable with large gallery
- support train and infer by setting a yaml file. [main.py](./main.py)
- a inference demo. [inference_api.py](./inference_api.py)


## run inference
```
# set the model path brefore running
python inference_api.py
```

- experimental results and trained models

|   | train dataset                                  | cnn-backbone | pooling | head   | classifier | performance on DukeMTMC-reID | model                                                                                                                                             |
|---|------------------------------------------------|--------------|---------|--------|------------|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | market, msmt, wildtrack, rap, njustwin         | resnet50     | avg     | bnhead | linear     | 0.6234(0.4196)               | [resnet50-market1501-msmt17-wildtrackcrop-rap-combineall-colorjitor-removerea-njust365win.pth](https://drive.google.com/file/d/1SvRSij1eqzEbgg6CWZbfbqbm6mFySnO3/view?usp=sharing)                                                      |
| 2 | market, msmt, wildtrack, rap, njustwin, cuhk03 | resnet50     | avg     | bnhead | circle     | 0.7410(0.5786)               | [market1501+msmt17+wildtrackcrop+rap(combineall)-colorjitor-removerea-njust365win-ibna-dataparallel-gpu2-p24k6-circle-60epochs-cuhk03-resnet50.pth](https://drive.google.com/file/d/1C-eUr3U-7LmR2i-YbC5Tgn_z3ywB4kXs/view?usp=sharing) |
|   |                                                |              |         |        |            |                              |                                                                                                                                                   |