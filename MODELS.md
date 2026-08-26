# Models

Per-model checkpoints, recommended serve settings, measured sweeps, and the
mistakes to avoid. All numbers measured on **6× Tesla V100-PCIE-32GB** (no NVLink,
all-PHB, PCIe Gen3 ×16) — see README.md for why that makes multi-GPU figures a
lower bound.

⚠️ **These cards are power-capped to 200 W against a 250 W default — but the cap is not
binding.** Measured under load: peak draw **83 W of 200 W** and SM clock at the **1380 MHz
hardware maximum**; a 250 W A/B changed nothing outside noise. The figures below are therefore
*not* understated for power reasons — see README.md → Test hardware.

Sweeps are aggregate tok/s at N concurrent streams, greedy (`temperature 0`)
unless noted. Launch everything with `./serve.sh`.

> **The `N ≤ 2` rule is FIXED (2026-08-26)** and no longer applies to any model below.
> The N=3 collapse was a cudagraph capture bug — 1Cat-vLLM pins the SM70 capture list
> to `[1,2]`, so every decode step with 3+ running requests dispatched
> `CUDAGraphMode.NONE` and ran fully eager. `serve.sh` now builds a capture ladder from
> `MNS` and `K` automatically: N=4 went 39.9 -> **297.7** tok/s (7.46x), N=16 205.7 ->
> **786.0**. See README -> Concurrency for the mechanism and the override knobs.
>
> ⚠️ **Every concurrency figure in the per-model sweeps below was measured before the
> fix** and understates N>=3 throughput. Single-stream numbers are unaffected.

---

## Why an A3B MoE gains far less than a dense model

The QPN kernels win by reading 4-bit weights instead of dequantizing to 16-bit. That only
helps when decode is **bandwidth-bound**, and a sparse MoE mostly is not:

| | weights read per token | ceiling @ 900 GB/s | measured @ N=1 | % of ceiling |
|---|---:|---:|---:|---:|
| dense 27B NVFP4 | 15.2 GB | ~59 tok/s | near ceiling | bandwidth-bound |
| **35B-A3B NVFP4** | **1.29 GB** | **~698 tok/s** | ~100 | **~14%** |

The A3B moves **11.8x less** weight data per token, so there is little bandwidth left to save.
What actually costs time is **960 tiny GEMMs per token** — 40 layers x 8 routed experts x 3
matrices, each only 2048x512 — which cannot fill an SM70 at batch 1. The workload is
launch-latency bound.

Two consequences, both measured and both counter to the usual MoE folklore:

- **More TP helps at N=1** (TP4 > TP2 > TP1) — extra SMs run those small GEMMs concurrently,
  and the allreduce is cheap at hidden_size 2048.
- **Batching helps enormously** — N=2 reaches 148.2 aggregate, because each expert GEMM
  finally has more than one row of work.

Expect a dense model's 2-3x from these kernels; expect roughly 20-30% on an A3B.

---

## Qwen3.6-35B-A3B — NVFP4, QPN2-MoE

[`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)

```bash
hf download nvidia/Qwen3.6-35B-A3B-NVFP4 --local-dir /models/nvidia/Qwen3.6-35B-A3B-NVFP4

MODEL=/models/nvidia/Qwen3.6-35B-A3B-NVFP4 SERVED=qwen36-35b \
GPU=0 TP=1 MML=32768 MNS=16 K=0 ./serve.sh
```

| N | 1 | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| aggTPS | **99.4** | **148.2** | 44.8 | 60.9 | 71.2 | 118.3 | 128.2 |

**99.4 tok/s single-stream for a 35B MoE on one 2017-era GPU.**

TP scaling (MML 32768, MNS 16) — **measured before the capture-ladder fix**, so every
TP appears to die at N=3. That was the `[1,2]` capture pin, which is TP-independent;
re-measure before citing these:

| N | TP1 | TP2 | TP4 |
|---|---|---|---|
| 1 | 92.7 | 97.5 | **100.2** |
| 2 | **148.2** | 146.4 | **159.7** |
| **3** | 22.4 | 27.8 | 23.9 |
| 4 | 28.4 | 31.9 | 35.7 |

### Do not

- ~~Do not run above N=2~~ **FIXED 2026-08-26.** The N=3 collapse was a cudagraph
  capture bug (`[1,2]` pin), not the serving path. With the capture ladder that
  `serve.sh` now builds automatically: N=3 **233.1** (was 39.8, 5.85x), N=4 **297.7**
  (was 39.9, 7.46x), N=16 **786.0** (was 205.7). Aggregate rises monotonically to
  N=16; per-stream tapers 95.1 -> 49.1. See README -> Concurrency.
- **Do not assume fewer TP ranks win.** On this model throughput rises monotonically
  with TP at N=1 — TP4 is fastest at N=1, N=2 *and* N=4. The "allreduce dominates for
  MoE" heuristic does not hold here.
- **Do not conclude the kernel causes the cliff.** With `MOEFLAG=0` (stock Marlin) the
  collapse is identical and per-stream throughput at N=3 lands within 4% (11.5 vs 11.1).
- **Do not enable MTP (`K>0`) on this checkpoint.** It is a large net *loss*, and the
  draft-acceptance rate will mislead you into keeping it:

  | K | tok/s @ N=1 | draft acceptance |
  |---|---|---|
  | **0** | **102.8** | — |
  | 2 | 37.3 | 92.3% |
  | 4 | 29.1 | 81.9% |

  92% acceptance and still 64% slower, and it is not a TP artefact — at TP4 the same pattern
  holds (K=0 **105.0** vs K=2 **41.1**, at 93.7% acceptance). The cause is in the weights: `nvidia/Qwen3.6-35B-A3B-NVFP4`
  quantizes the body to NVFP4 (`U8` packed + `F8_E4M3` scales) but ships the MTP head as **19 plain
  BF16 tensors**. Every draft step therefore leaves the QPN2/skinny path for the generic SM70
  unquantized-MoE kernel (`Using default MoE config. Performance might be sub-optimal`). A draft
  step reads 129 MB BF16 vs 36 MB if it were NVFP4 — and costs far more wall-time than even that
  implies, because decode here is latency-bound, not bandwidth-bound.

---

## Ornith-1.5-35B-A3B — NVFP4, QPN2-MoE

[`ornith-ai/Ornith-1.5-35B-A3B-NVFP4`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4)

The best all-round model on this stack — highest quality *and* competitive speed.

```bash
hf download ornith-ai/Ornith-1.5-35B-A3B-NVFP4 --local-dir /models/ornith-ai/Ornith-1.5-35B-A3B-NVFP4

# TP1 — fastest single-stream, 131K context
MODEL=/models/ornith-ai/Ornith-1.5-35B-A3B-NVFP4 SERVED=ornith15 \
GPU=4 TP=1 MML=131072 MNS=8 K=0 \
GENCFG='{"temperature":1.0,"top_p":0.95,"top_k":20,"frequency_penalty":0.3}' ./serve.sh

# TP2 — full 262K native context, 4.4x the concurrent capacity
MODEL=/models/ornith-ai/Ornith-1.5-35B-A3B-NVFP4 SERVED=ornith15 \
GPU=1,2 TP=2 MML=262144 MNS=16 K=0 \
GENCFG='{"temperature":1.0,"top_p":0.95,"top_k":20,"frequency_penalty":0.3}' ./serve.sh
```

**TP1, MML 131072, MNS 8** — card sampling `t=1.0 / top_p=0.95 / top_k=20`:

| sampling | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| no penalty | **99.5** | **142.1** | 37.8 | 60.2 |
| `frequency_penalty 0.3` | 93.7 | 137.0 | 42.6 | **70.3** |

**TP2, MML 262144 (full native), MNS 16, `frequency_penalty 0.3`:**

| run | N=1 | N=2 | N=4 | N=8 | N=16 |
|---|---|---|---|---|---|
| first | 99.3 | **150.1** | 36.4 | 63.0 | 119.5 |
| uncontended re-run | 98.6 | 139.3 | 30.7 | 51.8 | 110.0 |

**TP2 buys capacity, not speed:**

| | TP1 | TP2 |
|---|---|---|
| GPU KV pool | 425,598 tok | **1,869,960 tok** |
| concurrency @ 262K ctx | 1.62× | **7.13×** |
| single-stream decode | ~110.7 tok/s | ~100.5 tok/s |

~9% decode for **4.4× the concurrent full-context capacity**.

**Quality** (12-category rubric + SWE-bench Multilingual):

| metric | result |
|---|---|
| work categories (9) | **3.97 / 4.3** |
| medical categories (3) | **3.80 / 4.3** |
| SWE-bench Multilingual PHP-43 | **33/43**, 0 errors |

Of the 37 instances that produced a patch, 33 resolved (89%). The 10-instance gap was
6 empty patches (4 `LimitsExceeded`) — a capacity limit at TP1's 1.62× full-context
concurrency, not a reasoning failure.

### Do not

- **Do not use the DRY sampler.** It *causes* rep-1.000 collapse on code by fragmenting
  repetitive syntax. Use `frequency_penalty 0.3` instead — it is also worth +17% at N=8.
- ~~Do not run above N=2~~ **superseded 2026-08-26** by the capture-ladder fix (see
  README -> Concurrency). Concurrency now scales; TP2 remains the lever for context depth.
  The per-model sweeps below predate the fix and understate concurrent throughput.
- **Do not skip the reasoning cap.** The family has a documented runaway-reasoning trait.

---

## Laguna-S-2.1 — ModelOpt-NVFP4, QPN2-MoE

[`JasonW2025/Laguna-S-2.1-ModelOpt-NVFP4-W4A4-KVcal-vllm`](https://huggingface.co/JasonW2025/Laguna-S-2.1-ModelOpt-NVFP4-W4A4-KVcal-vllm) — safetensors, 64.4 GiB

**This is the cleanest kernel evidence in the repo, and a checkpoint you should not deploy.**

```bash
MODEL=/models/JasonW2025/Laguna-S-2.1-ModelOpt-NVFP4-W4A4-KVcal-vllm SERVED=laguna \
GPU=0,1,2,3 TP=4 MML=16384 MNS=1 K=0 ./serve.sh
```

Matched A/B against the stock path on identical hardware — TP4, greedy, 800 tokens, 3 runs:

| arm | tok/s | spread |
|---|---|---|
| stock Marlin-MoE (`MOEFLAG=0`) | 36.89 | 0.06 |
| **QPN2-MoE (`MOEFLAG=1`)** | **56.72** | 0.20 |
| **speedup** | **1.538×** | |

**Output was byte-identical between arms** — arithmetic, factual, code (77 tok) and prose
(104 tok) probes all matched exactly, with identical token counts. Route census: 36 ×
`qpn2_moe` at topk=10, zero fallbacks, QPN8 active. **A 1.54× speedup with provably
unchanged output is the strongest claim in this repository.**

KV scales unusually well — 36 of its 48 layers are sliding-window (w=512) and do not grow
with context; only the 12 full-attention layers pay per token:

| MML | KV pool | tokens | concurrency |
|---|---|---|---|
| 16,384 | 10.64 GiB | 503,693 | 30.74× |
| **131,072** | 10.64 GiB | **841,138** | 6.42× |

### Do not

- ⚠️ **Do not deploy this checkpoint.** Long-form greedy generation degenerates into
  repetition on **both** arms — the stock baseline is *worse* ("encompassing" ×544 vs 22
  repeated lines on the kernel). It quantizes 47 of 48 expert layers where the official
  repo leaves 40–47 unquantized. Use it to validate the kernel, not to serve users.
- **Do not read quality benchmarks into it.** The 12-category bench and SWE arm were
  deliberately not run: they would have measured checkpoint degradation, not the kernel.
- **Do not run it at TP1 and expect the TP>1 bug to appear.** The `supports_internal_mk`
  early-out bug (fixed in `fork_patches/modelopt.py`) is only reachable at TP>1.

---

## RadixArk Qwen3.8-27B-NVFP4-BF16-LMHead

NVFP4 W4A4 MLP (gs=16) + FP8 W8A8 attention + a **BF16 `lm_head`**, with a baked
`kv_cache_scheme: fp8`.

```bash
# TP2 is the efficient config; TP1 does not fit (see Do not)
MODEL=/models/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead SERVED=radixark \
GPU=0,1 TP=2 MML=32768 MNS=8 GMU=0.90 KVDTYPE=fp8_e4m3 \
LMHEAD=0 ./serve.sh
```

| N | TP4 agg | TP4 per-stream | TP2 agg | TP2 per-stream |
|---|---|---|---|---|
| 1 | 49.8 | 50.8 | 40.1 | 40.8 |
| **2** | **77.3** | 46.1 | **63.2** | 38.1 |
| 4 | 35.1 | **10.5** | 32.5 | **10.1** |
| 8 | 66.1 | 10.7 | 53.7 | 8.5 |

**Partition the array rather than scaling TP** — every config measured on the same box,
N=2 per instance:

| config | GPUs | combined tok/s | per GPU | vs 1× TP4 |
|---|---|---|---|---|
| 1× TP4 | 4 | 77.3 | 19.3 | 1.00× |
| 2× TP4 + MPS (**shared** cards) | 4 | 116.5 | 29.1 | 1.51× |
| 2× TP2 (**disjoint**) | 4 | 125.1 | 31.3 | 1.62× |
| **3× TP2 (disjoint)** | **6** | **187.1** | 31.2 | **2.42×** |

Disjoint pairs scale linearly — under 2.4% interference with three engines running flat out.

Memory: TP1 OOMs during **weight load**; runtime footprint is ~1.8× the 22.1 GiB checkpoint
(~40 GiB > 31.73 GiB per card). Non-KV per GPU doubles exactly as the GPU count halves
(10.2 → 19.3 GiB), i.e. weights shard cleanly.

### Do not

- ⚠️ **Do not leave `VLLM_SKINNY_LMHEAD=1`** (set `LMHEAD=0`). With it on, the loader finds
  no native NVFP4 lm_head and "packs from the model's own weights" — silently re-quantizing
  the BF16 head to 4 bits and discarding the only thing that distinguishes this checkpoint.
  **No error is raised.**
- **Do not try TP1.** It OOMs during weight load, before KV is even sized.
- **Do not co-locate two engines on shared cards if you can split the array** — sharing
  costs ~23% per instance even under MPS.

---

## Baselines on the same GPUs (not this fork)

For honesty about where QPN2-MoE actually wins.

| model | engine / quant | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|---|
| **Ornith-1.5-35B-A3B** | **QPN2-MoE NVFP4, TP1** | **99.5** | **142.1** | 37.8 | 60.2 |
| Ornith-1.5-35B-A3B | llama.cpp Q4_K_L, 1 GPU | 87.6 | 128.5 | — | — |
| Ornith-1.5-35B-A3B | llama.cpp Q8_0, 2 GPUs | 84.2 | 124.5 | — | — |
| Laguna-S-2.1 | llama.cpp UD-Q4_K_XL, 4 GPUs *(different quant)* | 51.4 | 59.1 | 100.9 | — |
| Gemma-4-31B | AWQ + TRITON_ATTN, TP2 | 38.9 | 56.9 | 110.0 | **156.3** |
| Qwen3.8-27B | FP8 (1cat-vllm), TP2 | 61.9 | 90.0 | 72.9 | 109.4 |
| Qwen3.8-27B | FP8 (1cat-vllm), TP4 | 74.2 | 102.3 | **181.1** | 131.2 |

**QPN2-MoE wins decisively at N=1–2** (99.5 vs 87.6 for the best llama.cpp alternative)
and **loses badly from N=4**, where AWQ/FP8 paths that scale monotonically overtake it.

Baseline checkpoints:

| model | repo | file |
|---|---|---|
| Ornith-1.5-35B-A3B Q4_K_L | [`bartowski/Ornith-1.5-35B-A3B-GGUF`](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF) | `Ornith-1.5-35B-A3B-Q4_K_L.gguf` |
| Ornith-1.5-35B-A3B Q8_0 | [`ornith-ai/Ornith-1.5-35B-A3B-GGUF`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) | `Ornith-1.5-35B-Q8_0.gguf` |
| Laguna-S-2.1 UD-Q4_K_XL | [`unsloth/Laguna-S-2.1-GGUF`](https://huggingface.co/unsloth/Laguna-S-2.1-GGUF) | `UD-Q4_K_XL/…-0000{1,2,3}-of-00003.gguf` |
| Gemma-4-31B-it AWQ | [`QuantTrio/gemma-4-31B-it-AWQ`](https://huggingface.co/QuantTrio/gemma-4-31B-it-AWQ) | safetensors |
| Qwen3.8-27B FP8 | [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | safetensors |

### Do not

- ⚠️ **Do not serve Gemma-4 on the default attention backend.** It needs
  `ATTN=TRITON_ATTN`. `FLASH_ATTN_V100` rejects it with
  `AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute` — Gemma-4 has
  heterogeneous head dims (256 / global 512) and the V100 XQA paged-decode kernel only
  supports head_dim=256. A Gemma/kernel interaction, not a Volta limitation.
- ⚠️ **Do not forget `--parallel N` divides `-c` in llama.cpp.** `-c 262144 --parallel 4`
  is 65,536 per slot, not 262,144. Sizing this wrong silently truncates context and reads
  as model failure.
