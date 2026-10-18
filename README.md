# Speculative KV Cache Quantization for Agentic Serving

CSE 585 course project (Fall 2026, University of Michigan).

**Team:** Arihan Iyer, Kushagra Mishra, Jatin Shihora, Rohit Puthucode

## Project summary

The KV cache grows linearly with context length, which is a serious memory
bottleneck for long-horizon, multi-turn agentic LLM workloads. Hard eviction
(H2O, SnapKV) permanently drops tokens that may matter later; uniform
quantization (KIVI) is safer but not content-aware.

We propose a **speculative multi-rollout KV cache quantization** scheme:

1. When GPU memory pressure crosses a threshold, fork the current context to
   a lightweight draft model.
2. Run `N` parallel stochastic draft rollouts (`N ∈ [4, 8]`, lookahead
   `L = 10`) to sample diverse plausible continuations.
3. Score each context token by **consensus attention** across rollouts:
   `S_i = (1/N) * Σ_j max_t A^(j)_{t,i}`.
4. Keep high-consensus tokens (plus attention sinks / recent window) in
   FP16/BF16; quantize the rest to 4-bit or 2-bit.


## Repo layout

```
.
├── proposal/
│   ├── proposal.tex        # proposal
│   └── references.bib      # bibliography stub — fill in real fields
├── src/
│   ├── baseline_experiment.py   # Phase 0: profile vanilla HF KV cache under simulated agentic load
│   └── plot_baseline.py         # plots memory/latency vs. context length from the baseline CSV
├── data/
│   └── sample_tool_outputs.jsonl  # example trace schema — replace with real SWE-bench/BFCL traces
├── results/                # baseline_experiment.py output lands here (gitignored)
└── requirements.txt
```

## Phase 0: baseline

Profiles the default HuggingFace `transformers` KV cache (no optimizations)
under a simulated agentic loop: each turn appends a tool-output chunk to the
growing context (reusing `past_key_values` across turns), then generates the
next step. Logs per-turn context length, prefill/decode latency, and GPU
memory to CSV.

```bash
pip install -r requirements.txt

# smoke test with synthetic filler text — NOT for reported results
python src/baseline_experiment.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --num-turns 5 --gen-len 16


!python src/baseline_experiment.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --tool-outputs data/real_single_task.jsonl \
    --num-turns 126 --gen-len 64 --out results/baseline_real.csv


python src/plot_baseline.py --csv results/baseline.csv --out results/baseline_plot.png
```



