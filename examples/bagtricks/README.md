# BagTrick

implement and speedup [BagTricks](https://arxiv.org/abs/1903.07071) with light-reid

## Experimental Results and Trained Models

DukeMTMC-reID

<table><thead><tr><th rowspan="2">DukeMTMC<br>-ReID</th><th colspan="3">light-reid</th><th colspan="4">performance</th><th colspan="2">time(on a TITAN XP)</th></tr><tr><td>model<br>(res18)</td><td>light<br>feature</td><td>light<br>search</td><td>feature<br>dim</td><td>metric</td><td>R1</td><td>mAP</td><td>inference<br>per batch(64)</td><td>search<br>per query</td></tr></thead><tbody><tr><td>1</td><td>-</td><td>-</td><td>-</td><td>2048</td><td>cosine</td><td>0.870</td><td>0.772</td><td>78.6ms</td><td>154.0ms</td></tr><tr><td>2</td><td>√</td><td>-</td><td>-</td><td>512</td><td>cosine</td><td>0.866</td><td>0.751</td><td>25.9ms</td><td>34.7ms</td></tr><tr><td>3</td><td>-</td><td>√</td><td>-</td><td>2048</td><td>hamming</td><td>0.872</td><td>0.768</td><td>77.3ms</td><td>23.1ms</td></tr><tr><td>4</td><td>-</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td>0.865</td><td>0.728</td><td>75.3ms</td><td>12.1ms</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td>0.856</td><td>0.714</td><td>23.2ms</td><td>14.2ms</td></tr></tbody></table>

Market-1501

<table><thead><tr><th rowspan="2">Market-1501</th><th colspan="3">light-reid</th><th colspan="4">performance</th><th colspan="2">time(on a TITAN XP)</th></tr><tr><td>model<br>(res18)</td><td>light<br>feature</td><td>light<br>search</td><td>feature<br>dim</td><td>metric</td><td>R1</td><td>mAP</td><td>inference<br>per batch(64)</td><td>search<br>per query</td></tr></thead><tbody><tr><td>1</td><td>-</td><td>-</td><td>-</td><td>2048</td><td>cosine</td><td>0.937</td><td>0.856</td><td>78.6ms</td><td>246.0ms</td></tr><tr><td>2</td><td>√</td><td>-</td><td>-</td><td>512</td><td>cosine</td><td>0.932</td><td>0.835</td><td>22.7ms</td><td>68.7ms</td></tr><tr><td>3</td><td>-</td><td>√</td><td>-</td><td>2048</td><td>hamming</td><td>0.939</td><td>0.851</td><td>73.7ms</td><td>32.3ms</td></tr><tr><td>4</td><td>-</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td>0.929</td><td>0.836</td><td>75.1ms</td><td>17.1ms</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td></td><td></td><td></td><td></td></tr></tbody></table>

MSMT17

<table><thead><tr><th rowspan="2">MSMT17</th><th colspan="3">light-reid</th><th colspan="4">performance</th><th colspan="2">time(on a TITAN XP)</th></tr><tr><td>model<br>(res18)</td><td>light<br>feature</td><td>light<br>search</td><td>feature<br>dim</td><td>metric</td><td>R1</td><td>mAP</td><td>inference<br>per batch(64)</td><td>search<br>per query</td></tr></thead><tbody><tr><td>1</td><td>-</td><td>-</td><td>-</td><td>2048</td><td>cosine</td><td>0.740</td><td>0.516</td><td>73.1ms</td><td></td></tr><tr><td>2</td><td>√</td><td>-</td><td>-</td><td>512</td><td>cosine</td><td>0.719</td><td>0.478</td><td>24.2ms</td><td></td></tr><tr><td>3</td><td>-</td><td>√</td><td>-</td><td>2048</td><td>hamming</td><td></td><td></td><td></td><td></td></tr><tr><td>4</td><td>-</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td></td><td></td><td></td><td></td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>multiple</td><td>hamming</td><td></td><td></td><td></td><td></td></tr></tbody></table>
