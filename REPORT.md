# Prefix-Cache Invalidation Benchmark Report

**Run:** 2026-11-20 UTC  
**Model:** `Qwen/Qwen2.5-14B-Instruct`  
**Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB VRAM, driver 595.91.07, CUDA 13.2  
**Software:** vLLM 0.34.0, torch 2.14.0+cu130, transformers 5.22.2, httpx 0.28.1  
**Server:** vLLM `OpenAI API server` with prefix caching enabled, block size 16, `--max-model-len 32768`, `--gpu-memory-utilization 0.9`; KV cache 759,296 tokens, max concurrency 23.17 requests at max_model_len.

## Methodology

The runner (`benchmark_prefix_vs_rewrite.py`) tests three arms, each as a 50-sample (10 scenarios x 5 reps) paired experiment, with a unique session salt per trial so cached KV blocks never leak between trials:

- `append_only` — Turn-1 context is primed, then the measured request appends a new tool result without modifying any earlier prefix.
- `rewrite_equal_tokens` — an early summary is rewritten; measured total prompt tokens are equal to `append_only` within +/-2 tokens.
- `rewrite_shorter_prompt` — an early summary is rewritten shorter; measured total prompt tokens are tuned to ~0.70x of `append_only`.

Execution details:

- 10 scenarios x 5 reps x 3 arms = **150 measured requests**; plus 15 warmup requests, discarded.
- Concurrency = 1, temperature = 0.0, measured `max_tokens = 512`.
- Turn-1 priming target = 5,500 tokens; tool output target = 2,200 tokens.
- Token parity was enforced with the exact HF tokenizer: `parity_max_abs_diff = 0.0` across all 50 pairs; shorter-prompt ratio mean 0.699997, range 0.699936-0.700064.
- Data: `artifacts/vm_results/results/raw_trials.jsonl`, `summary_metrics.json`, `summary_metrics.csv`, `server_telemetry.json`, `serve_config.json`, `ttft_comparison.png`, `uncached_tokens_vs_ttft.png`, `shorter_prompt_paradox.png`.

A discarded pilot run at the original 128-token cap is not used for inference below; the corrected run is the official result.

## Results

Measured per arm (n = 50 each). Values are means; median and p95 in parentheses.

| Arm | Total input tokens | Cached tokens | Uncached tokens | TTFT ms (p50, p95) | E2E ms | ITL ms | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| `append_only` | 7761.6 (7761.0, 7785.0) | 5488.0 | 2273.6 | 40.12 (39.24, 44.42) | 369.24 | 2.30 | 84.0% |
| `rewrite_equal_tokens` | 7761.6 (7761.0, 7785.0) | 212.8 | 7548.8 | 68.04 (67.29, 73.35) | 359.75 | 2.29 | 74.0% |
| `rewrite_shorter_prompt` | 5433.1 (5433.0, 5450.0) | 213.76 | 5219.34 | 43.89 (43.18, 47.44) | 381.71 | 2.14 | 80.0% |

### Paired deltas and interpretation

| Pair | Uncached delta (B/C - A) | Cached ratio to A | TTFT delta (mean, 95% CI) |
|---|---:|---:|---:|
| `rewrite_equal_tokens` vs `append_only` | +5275.2 tokens | 0.0388 | +27.91 ms ([27.20, 28.55]) |
| `rewrite_shorter_prompt` vs `append_only` | +2945.74 tokens | 0.0389 | +3.76 ms ([3.10, 4.31]) |

- Rewriting the early summary invalidates ~5,275 more uncached tokens in the equal-token arm: the mean TTFT penalty is **+27.9 ms (ratio 1.70x)**; the predicted slope cost from the run-level regression is 26.999 ms, with residual +0.915 ms.
- The shorter rewrite still arrives ~+3.8 ms later than `append_only` despite sending only ~2,946 more tokens than `append_only` and being 2.30x A's uncached-token count. The short paradox is confirmed but small on this GPU.
- Observed total-token ratio for `rewrite_shorter_prompt` is 0.699997 of `append_only` (protocol target 0.70). The runner's break-even estimate is `r*=0.3204`: below about 32% of A's total tokens the appended-tool path is predicted to become favorable; at the observed 70%, rewriting is still slower by ~3.8 ms.
- Cache-hit fractions: `append_only` 0.707, `rewrite_equal_tokens` 0.027, `rewrite_shorter_prompt` 0.039.

### Run-level regression

Client-measured TTFT vs uncached prompt tokens (n = 150): `TTFT ≈ 25.02 + 5.118 * uncached_tokens` ms; R² = 0.773. This is consistent with the paired B-A slope prediction above.

### Telemetry

- GPU memory: 97,887 MiB total; 86,151 MiB used at run start; 86,575 MiB used at end.
- Temperature 30°C -> 43°C; power draw 85 W -> 204.64 W; SM clocks 2370 -> 2415 MHz.
- Engine/server metrics confirm prefix caching is active and that measured usage and Prometheus cached-token counters agree (`cached_tokens_usage` vs `cached_tokens_metrics` mean absolute diff 0.0; correlation 1.0).
- Preemptions: 0.

## Reproducing

On the VM (or locally after `pip install`), run:

```bash
HF_HOME=/home/ubuntu/hf /home/ubuntu/venv/bin/python benchmark_prefix_vs_rewrite.py --config run.yaml
```

The interactive notebook with the same analysis is `prefill_cache_benchmark.ipynb`.
