# light-reid Model Zoo

This file collect a set of pre-train models.


## run

```
# train
python train.py --config_file ./base_config.yaml
``` 

```
# infer
python infer.py --config_file ./base_config.yaml --model_path /path/to/model.pth
```


## Multiple-Datasets Training

Training a [model](https://drive.google.com/file/d/1hkNGv__e4zCWPri8L1S_l_Y1BoK-6wlP/view?usp=sharing) with multiple datasets,
makes it stronger and more generalizable

<table><thead><tr><th>Training Dataset</th><th>Test Dataset</th><th>       map</th><th>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;rank-1</th><th>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;rank-5</th><th>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;rank-10</th></tr></thead><tbody><tr><td rowspan="4">DukeMTMC-reID<br>Market-1501<br>MSMT17<br>CUHK03<br>WildTrack<br>NJUST<br>RAP(private)</td><td>DukeMTMCreID</td><td>0.6824988661464477</td><td>0.835278276481149</td><td>0.914721723518851</td><td>0.9349192100538599</td></tr><tr><td>Market1501</td><td>0.7929966369451702</td><td>0.9263657957244655</td><td>0.9756532066508313</td><td>0.9842636579572447</td></tr><tr><td>MSMT17</td><td>0.3968708072353888</td><td>0.7006604339994854</td><td>0.8137061497555537</td><td>0.8516167767389999</td></tr><tr><td>CUHK03</td><td>0.694728373609779</td><td>0.7307142857142858</td><td>0.875</td><td>0.9235714285714286</td></tr></tbody></table>

