# BagTrick(build with config files)

implement and speedup [BagTricks](https://arxiv.org/abs/1903.07071) with light-reid

build with config files, more simple

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

<table><thead><tr><th rowspan="2">DukeMTMC<br>-ReID<br>(gallery size: 17661)</th><th colspan="3">light-reid</th><th colspan="4">performance</th><th colspan="2">time(on a TITAN XP)</th></tr><tr><td>model<br>(res18)</td><td>light<br>feature</td><td>light<br>search</td><td>feature<br>dim</td><td>metric</td><td>R1</td><td>mAP</td><td>inference<br>per batch(64)</td><td>search<br>per query</td></tr></thead><tbody><tr><td>1</td><td>-</td><td>-</td><td>-</td><td>2048</td><td>cosine</td><td>0.870</td><td>0.772</td><td>78.6ms</td><td>237.1ms</td></tr><tr><td>2</td><td>√</td><td>-</td><td>-</td><td>512</td><td>cosine</td><td>0.866</td><td>0.751</td><td>25.9ms</td><td>46.7ms</td></tr><tr><td>3</td><td>-</td><td>√</td><td>-</td><td>2048</td><td>hamming</td><td>0.872</td><td>0.768</td><td>77.3ms</td><td>73.1ms</td></tr><tr><td>4</td><td>-</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td>0.865</td><td>0.728</td><td>75.3ms</td><td>15.1ms</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td>0.856</td><td>0.714</td><td>23.2ms</td><td>16.0ms</td></tr></tbody></table>

Market-1501

<table><thead><tr><th rowspan="2">Market-1501<br>(gallery size: 19732)</th><th colspan="3">light-reid</th><th colspan="4">performance</th><th colspan="2">time(on a TITAN XP)</th></tr><tr><td>model<br>(res18)</td><td>light<br>feature</td><td>light<br>search</td><td>feature<br>dim</td><td>metric</td><td>R1</td><td>mAP</td><td>inference<br>per batch(64)</td><td>search<br>per query</td></tr></thead><tbody><tr><td>1</td><td>-</td><td>-</td><td>-</td><td>2048</td><td>cosine</td><td>0.937</td><td>0.856</td><td>78.6ms</td><td>382.0ms</td></tr><tr><td>2</td><td>√</td><td>-</td><td>-</td><td>512</td><td>cosine</td><td>0.932</td><td>0.835</td><td>22.7ms</td><td>58.7ms</td></tr><tr><td>3</td><td>-</td><td>√</td><td>-</td><td>2048</td><td>hamming</td><td>0.939</td><td>0.851</td><td>73.7ms</td><td>83.2ms</td></tr><tr><td>4</td><td>-</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td>0.929</td><td>0.836</td><td>75.1ms</td><td>17.1ms</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td>0.922</td><td>0.815</td><td>22.2ms</td><td>17.7ms</td></tr></tbody></table>

MSMT17

<table><thead><tr><th rowspan="2">MSMT17<br>(gallery size: 82161)</th><th colspan="3">light-reid</th><th colspan="4">performance</th><th colspan="2">time(on a TITAN XP)</th></tr><tr><td>model<br>(res18)</td><td>light<br>feature</td><td>light<br>search</td><td>feature<br>dim</td><td>metric</td><td>R1</td><td>mAP</td><td>inference<br>per batch(64)</td><td>search<br>per query</td></tr></thead><tbody><tr><td>1</td><td>-</td><td>-</td><td>-</td><td>2048</td><td>cosine</td><td>0.740</td><td>0.516</td><td>73.1ms</td><td>1364.5ms</td></tr><tr><td>2</td><td>√</td><td>-</td><td>-</td><td>512</td><td>cosine</td><td>0.719</td><td>0.478</td><td>24.2ms</td><td>312.4ms</td></tr><tr><td>3</td><td>-</td><td>√</td><td>-</td><td>2048</td><td>hamming</td><td>0.760</td><td>0.533</td><td>76.2ms</td><td>371.3ms</td></tr><tr><td>4</td><td>-</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td>0.739</td><td>0.493</td><td>75.2ms</td><td>95.0ms</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td></td><td></td><td></td><td></td></tr></tbody></table>
