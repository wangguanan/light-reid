# AGW

implement and speedup [AGW](https://arxiv.org/abs/2001.04193) with light-reid.

## Experimental Results and Trained Models

Settings (on a MacBook Pro (Retina, 13-inch, Mid 2014))

- GPU: TITAN XP (memory 12194MB)
- CPU: 2.6 GHz Dual-Core Intel Core i5
- Memory: 8 GB 1600 MHz DDR3

DukeMTMC-reID

<table><thead><tr><th rowspan="2">DukeMTMC-reID</th><th colspan="3">light-reid</th><th colspan="4">performance</th><th colspan="2">time(on a TITAN XP)</th></tr><tr><td>model<br>(res18)</td><td>light<br>feature</td><td>light<br>search</td><td>feature<br>dim</td><td>metric</td><td>R1</td><td>mAP</td><td>inference<br>per batch(64)</td><td>search<br>per query</td></tr></thead><tbody><tr><td>1</td><td>-</td><td>-</td><td>-</td><td>2048</td><td>cosine</td><td>0.886</td><td>0.786</td><td>78.6ms</td><td>248.3ms</td></tr><tr><td>2</td><td>√</td><td>-</td><td>-</td><td>512</td><td>cosine</td><td>0.856</td><td>0.743</td><td>23.2ms</td><td>61.5ms</td></tr><tr><td>3</td><td>-</td><td>√</td><td>-</td><td>2048</td><td>hamming</td><td>0.875</td><td>0.776</td><td>78.2ms</td><td>30.2ms</td></tr><tr><td>4</td><td>-</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td>0.878</td><td>0.758</td><td>73.9ms</td><td>20.0ms</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td>0.851</td><td>0.722</td><td>23.7ms</td><td></td></tr></tbody></table>

Market-1501

<table><thead><tr><th rowspan="2">Market-1501<br>(gallery size: 19732)</th><th colspan="3">light-reid</th><th colspan="4">performance</th><th colspan="2">time(on a TITAN XP)</th></tr><tr><td>model<br>(res18)</td><td>light<br>feature</td><td>light<br>search</td><td>feature<br>dim</td><td>metric</td><td>R1</td><td>mAP</td><td>inference<br>per batch(64)</td><td>search<br>per query</td></tr></thead><tbody><tr><td>1</td><td>-</td><td>-</td><td>-</td><td>2048</td><td>cosine</td><td>0.945</td><td>0.865</td><td>76.3ms</td><td></td></tr><tr><td>2</td><td>√</td><td>-</td><td>-</td><td>512</td><td>cosine</td><td>0.931</td><td>0.836</td><td>22.7ms</td><td></td></tr><tr><td>3</td><td>-</td><td>√</td><td>-</td><td>2048</td><td>hamming</td><td>0.942</td><td>0.859</td><td>78.2ms</td><td></td></tr><tr><td>4</td><td>-</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td>0.942</td><td>0.852</td><td>75.3ms</td><td></td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td>0.922</td><td>0.809</td><td>22.0ms</td><td></td></tr></tbody></table>

MSMT17

<table><thead><tr><th rowspan="2">MSMT17<br>(gallery size: 82161)</th><th colspan="3">light-reid</th><th colspan="4">performance</th><th colspan="2">time(on a TITAN XP)</th></tr><tr><td>model<br>(res18)</td><td>light<br>feature</td><td>light<br>search</td><td>feature<br>dim</td><td>metric</td><td>R1</td><td>mAP</td><td>inference<br>per batch(64)</td><td>search<br>per query</td></tr></thead><tbody><tr><td>1</td><td>-</td><td>-</td><td>-</td><td>2048</td><td>cosine</td><td>0.757</td><td>0.528</td><td>74.0ms</td><td></td></tr><tr><td>2</td><td>√</td><td>-</td><td>-</td><td>512</td><td>cosine</td><td>0.729</td><td>0.489</td><td>23.4ms</td><td></td></tr><tr><td>3</td><td>-</td><td>√</td><td>-</td><td>2048</td><td>hamming</td><td>0.768</td><td>0.539</td><td>78.6ms</td><td></td></tr><tr><td>4</td><td>-</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td>0.762</td><td>0.519</td><td>76.3ms</td><td></td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td>0.713</td><td>0.462</td><td>21.7ms</td><td></td></tr></tbody></table>
