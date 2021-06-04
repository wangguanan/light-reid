# BagTrick with PCA reduction

reduce feature dimension with PCA, obtaining shorter feature meanwhile keep strong accuracy.

## run

```
# train
python train.py --config_file ./base_config.yaml
``` 

```
# infer
python infer.py --config_file ./base_config.yaml --model_path /path/to/model.pth
```


## Experimental Results and Trained Models

Settings (on a MacBook Pro (Retina, 13-inch, Mid 2014))
- GPU: TITAN XP (memory 12194MB)
- CPU: 2.6 GHz Dual-Core Intel Core i5
- Memory: 8 GB 1600 MHz DDR3

DukeMTMC-reID

<table><thead><tr><th>DukeMTMC<br>-ReID<br>(gallery size: 17661)</th><th colspan="4">light-reid</th><th colspan="5">performance</th><th colspan="2">time(on a TITAN XP)</th></tr></thead><tbody><tr><td></td><td>light<br>model</td><td>light<br>feature</td><td>light<br>search</td><td>reduction</td><td>CNN</td><td>feature<br>dim</td><td>metric</td><td>R1</td><td>mAP</td><td>inference<br>per batch(64)</td><td>search<br>per query</td></tr><tr><td>1</td><td>-</td><td>-</td><td>-</td><td>none</td><td>ResNet50</td><td>2048</td><td>cosine</td><td>0.870</td><td>0.772</td><td>78.6ms</td><td>237.1ms</td></tr><tr><td>2</td><td>-</td><td>-</td><td>-</td><td>pca-128</td><td>ResNet50</td><td>128</td><td>cosine</td><td>0.863</td><td>0.752</td><td>78.6ms</td><td>20.7ms</td></tr></tbody></table>

Market-1501

- comming soon

MSMT17

- comming soon

