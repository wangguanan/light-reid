# AGW

implement and speedup [AGW](https://arxiv.org/abs/2001.04193) with light-reid.

## Experimental Results and Trained Models

<table><thead><tr><th rowspan="2">DukeMTMC-reID</th><th colspan="3">light-reid</th><th colspan="4">performance</th><th colspan="2">time(on a TITAN XP)</th></tr><tr><td>model<br>(res18)</td><td>light<br>feature</td><td>light<br>search</td><td>feature<br>dim</td><td>metric</td><td>R1</td><td>mAP</td><td>inference<br>per batch(64)</td><td>search<br>per query</td></tr></thead><tbody><tr><td>1</td><td>-</td><td>-</td><td>-</td><td>2048</td><td>cosine</td><td>0.886</td><td>0.786</td><td>78.6ms</td><td>248.3ms</td></tr><tr><td>2</td><td>√</td><td>-</td><td>-</td><td>512</td><td>cosine</td><td>0.856</td><td>0.743</td><td>23.2ms</td><td>61.5ms</td></tr><tr><td>3</td><td>-</td><td>√</td><td>-</td><td>2048</td><td>hamming</td><td>0.875</td><td>0.776</td><td>78.2ms</td><td>30.2 ms</td></tr><tr><td>4</td><td>-</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td></td><td></td><td></td><td></td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td></td><td></td><td></td><td></td></tr></tbody></table>