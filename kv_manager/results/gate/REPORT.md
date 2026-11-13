# Phase 2 gate: results

**Verdict: NO-GO.** No draft-guided configuration passes the pre-registered rule at either block size.
Full tables are in [summary.md](summary.md), per-row data in `metrics.csv` (31 samples × 2 block sizes × 51 schemes),
plots in `pareto_B16.png` and `pareto_B64.png`, and the full job log in `run.log`.

## Headline numbers (31 pauses, histories of 8.0k–33.8k tokens)

| Block 16 | bytes | ΔNLL (nats/token) | top-1 |
|---|---|---|---|
| uniform_bf16 (control) | 1.000 | exactly 0 on every sample | 1.000 |
| **uniform_int4** | **0.325** | **+0.0009** | **0.991** |
| uniform_int2 | 0.200 | +0.0332 | — |
| consensus, top 10% bf16 + rest int2 | 0.280 | +0.0091 | 0.975 |
| consensus, top 40% int4 + rest int2 | 0.250 | +0.0057 | 0.988 |
| specprefill, top 10% bf16 | 0.280 | +0.0079 | 0.979 |
| oracle, top 10% bf16 | 0.280 | +0.0154 | 0.979 |
| random, top 10% bf16 | 0.280 | +0.0296 | 0.963 |

With block size 64 every draft configuration does worse than with 16; uniform_int4 there is +0.0035 at 0.282 bytes.

## What it means

1. **Uniform 4-bit is almost lossless for the 8B target**: +0.001 nats/token, 99.1% top-1 agreement, at 0.325
   of the bf16 bytes. It leaves the draft almost nothing to win.
2. **The limit is the 2-bit floor, not the scorer.** Even the oracle (the target's own attention from the real next
   message) never beats uniform_int4: every oracle configuration's paired CI is entirely above 0. Any scheme that
   puts most blocks at int2 pays more than int4 costs everywhere.
3. **The draft scorers do real work, they just can't overcome (2).** At keep 10%: recall vs oracle is 0.58 for
   consensus vs 0.10 for random and 0.22 for recency, and ΔNLL is roughly a third of random's. Consensus and SpecPrefill
   scoring are statistically indistinguishable (every paired CI includes 0). The draft scorers even beat the oracle on
   ΔNLL: the max-over-next-message attention isn't the best proxy for sensitivity to quantization.
4. **Closest miss:** draft_window, top 40% int4, B=16: 0.250 bytes (rule (b) needs ≤ 0.244) and CI upper bound
   +0.0118 (needs < +0.01). It fails both conditions, narrowly.
5. **Draft scoring latency** (A40, HF eager, median over samples, history median 16.6k tokens): draft_window 81 ms,
   specprefill 246 ms, consensus 701 ms.

Next step per [docs/04-phases.md](../../../docs/04-phases.md): the fallback, uniform low-bit reload (M1).
Possibly worth a look later: int4 for the bulk with bf16 for the top blocks, which this grid did not include.

## Deviations from the documented protocol

| Item | Documented | This run | Effect |
|---|---|---|---|
| Reference NLL | taken from the oracle pass (eager attention) | sdpa, same kernel as every scheme; oracle in a separate pass | **Bug fix.** With identical bf16 KV, the old code reported up to +0.010 ΔNLL for a lossless reload. The verdict doesn't depend on it: the CI tests use paired differences, which the offset cancels out of. |
| Control | none | `uniform_bf16` row added; ΔNLL exactly 0 everywhere | sanity check only, not in the verdict |
| Quantized copies | all layers held at once | made per layer on the fly | bit-identical results (checked on 102 rows); needed to fit 48 GB |
| GPU | A100 80 GB | NVIDIA A40 48 GB (driver 595.71), UMich Great Lakes | latencies are A40 numbers |
| Environment | lock file (torch 2.13.0 cu130, vLLM 0.27.1) | torch 2.14.1+cu126, transformers 5.14.1, no vLLM | possible tiny numeric differences |
| Data download | `data.py` | same URLs fetched with curl, converted by `data.py` | none (same 31 samples) |

All other settings are the code defaults: models, hist lens 8k/16k/24k/32k, keep 5/10/20/40%, hi ∈ {bf16, int4},
B ∈ {16, 64}, N=4, L=10, window 32, SpecPrefill lookahead 8, seed 0. Wall time was about 40 minutes.
