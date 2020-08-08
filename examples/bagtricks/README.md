# BagTrick

implement and speedup BagTricks with light-reid

## Experimental Results and Trained Models

<table><thead><tr><th rowspan="2">DukeMTMC-reID</th><th colspan="3">light-reid</th><th colspan="2">performance</th><th colspan="2">time(on a TITAN XP)</th></tr><tr><td>light-model(res18)</td><td>light-feature</td><td>light-search</td><td>rank-1</td><td>mAP</td><td>inference/batch(64)</td><td>search/query</td></tr></thead><tbody><tr><td>1</td><td>-</td><td>-</td><td>-</td><td>0.869</td><td>0.772</td><td>0.01516s</td><td>174.6ms</td></tr><tr><td>2</td><td>√</td><td>-</td><td>-</td><td>0.866</td><td>0.751</td><td>0.00839s</td><td></td></tr><tr><td>3</td><td>-</td><td>√</td><td>-</td><td></td><td></td><td></td><td></td></tr><tr><td>4</td><td>-</td><td>-</td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td>5</td><td>-</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td>6</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td></tr></tbody></table>