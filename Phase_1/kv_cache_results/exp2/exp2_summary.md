Contexts evaluated: 12

Everything-2-bit baseline (uniform_2bit): Δ NLL = +0.054

### Δ NLL (lower is better), mean over contexts

| Method | Policy | keep 5% | keep 10% | keep 20% | keep 40% |
|---|---|---|---|---|---|
| consensus_N4 | quant | +0.058 | +0.043 | +0.007 | -0.008 |
| greedy_N1 | quant | +0.058 | +0.032 | +0.026 | +0.010 |
| snapkv | quant | +0.047 | +0.028 | +0.033 | +0.035 |
| random | quant | +0.083 | +0.033 | +0.056 | +0.038 |
| oracle | quant | +0.051 | +0.030 | +0.002 | +0.006 |
| consensus_N1 | quant | +0.037 | +0.014 | +0.023 | -0.000 |
| consensus_N2 | quant | +0.068 | +0.042 | +0.029 | -0.004 |
| consensus_N8 | quant | +0.068 | +0.022 | +0.017 | -0.006 |
| consensus_N4 | evict | +0.496 | +0.317 | +0.123 | +0.060 |
| greedy_N1 | evict | +0.609 | +0.337 | +0.173 | +0.054 |
| snapkv | evict | +0.928 | +0.628 | +0.333 | +0.148 |
| random | evict | +1.110 | +0.908 | +0.510 | +0.232 |
| oracle | evict | -0.005 | +0.001 | +0.002 | -0.003 |
| consensus_N1 | evict | +0.519 | +0.323 | +0.142 | +0.057 |
| consensus_N2 | evict | +0.564 | +0.333 | +0.116 | +0.061 |
| consensus_N8 | evict | +0.475 | +0.292 | +0.100 | +0.061 |

### Top-1 agreement with the uncompressed model (higher is better)

| Method | Policy | keep 5% | keep 10% | keep 20% | keep 40% |
|---|---|---|---|---|---|
| consensus_N4 | quant | 93.5% | 92.9% | 95.2% | 97.3% |
| greedy_N1 | quant | 92.3% | 94.1% | 95.4% | 96.1% |
| snapkv | quant | 91.8% | 91.5% | 92.7% | 94.8% |
| random | quant | 91.6% | 90.2% | 91.7% | 91.7% |
| oracle | quant | 93.0% | 95.2% | 96.7% | 98.2% |
| consensus_N1 | quant | 93.1% | 93.8% | 95.3% | 96.0% |
| consensus_N2 | quant | 92.1% | 95.2% | 96.3% | 96.9% |
| consensus_N8 | quant | 92.5% | 93.2% | 96.3% | 96.7% |
| consensus_N4 | evict | 80.7% | 85.6% | 92.4% | 96.8% |
| greedy_N1 | evict | 78.9% | 85.9% | 92.4% | 95.7% |
| snapkv | evict | 72.6% | 79.4% | 85.9% | 92.9% |
| random | evict | 65.5% | 68.9% | 76.3% | 86.5% |
| oracle | evict | 92.9% | 93.8% | 96.2% | 98.2% |
| consensus_N1 | evict | 79.8% | 86.7% | 91.9% | 97.0% |
| consensus_N2 | evict | 79.3% | 87.7% | 92.7% | 96.8% |
| consensus_N8 | evict | 78.7% | 86.2% | 92.9% | 96.4% |

### Recall of the oracle's top tokens

| Method | top 5% | top 10% | top 20% | top 40% |
|---|---|---|---|---|
| consensus_N4 | 43.4% | 52.3% | 64.4% | 75.3% |
| greedy_N1 | 42.2% | 50.4% | 61.6% | 73.3% |
| snapkv | 21.5% | 32.2% | 47.7% | 63.4% |
| random | 4.8% | 9.5% | 19.9% | 40.4% |
| consensus_N1 | 41.5% | 51.2% | 62.4% | 73.9% |
| consensus_N2 | 41.2% | 51.3% | 62.9% | 74.1% |
| consensus_N8 | 43.7% | 52.9% | 65.7% | 76.8% |

### Do the guesses agree? (overlap of top-10% token sets, Jaccard)

- Between two different sampled guesses: 0.57
- Between consensus N=4 and the single greedy guess: 0.66
