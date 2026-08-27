# volta-nvfp4

**NVFP4 Mixture-of-Experts inference on Tesla V100 (SM70).**

A fork of [**v100-skinny**](https://github.com/dnv2003/v100-skinny) by **Dennis Vertzyas**,
extending its QPN2 quantized-projection kernels to handle **fused MoE expert stacks**
(`switch_mlp`), so NVFP4 MoE models run on Volta — hardware with no native FP4 or FP8 support.

Upstream made dense NVFP4 work on SM70. This fork makes **MoE** work.

> All credit for the QPN2/QPN8 kernels, the `mma.sync.m8n8k4` approach, and the entire SM70
> NVFP4 foundation goes to v100-skinny. This repository is a derivative — please cite upstream,
> see [`CITATION.cff`](CITATION.cff).

Derivation chain: `vLLM → 1Cat-vLLM (SM70 fork) → v100-skinny → volta-nvfp4`

---

## Headline result

A **1.538× speedup with byte-identical output**, measured as a matched A/B against the stock
path on the same GPUs (Laguna-S-2.1, TP4, 3 runs):

| arm | tok/s |
|---|---|
| stock Marlin-MoE | 36.89 |
| **QPN2-MoE** | **56.72** |

And **99.4 tok/s single-stream for a 35B MoE on one 2017-era GPU** (Qwen3.6-35B-A3B).

## Concurrency: the `N ≤ 2` rule is FIXED (2026-08-26)

**Earlier releases of this repo told you to cap at N=2.** That limit was never a property
of the hardware or the kernels — it was a cudagraph capture bug, and it is now fixed. On
the same two V100s, same NVFP4 weights, same QPN2-MoE kernels:

| N | before (capture `[1,2]`) | after (capture ladder) | speedup |
|---|---:|---:|---:|
| 1 | 95.4 | 95.1 | 1.00x |
| 2 | 160.8 | 172.3 | 1.07x |
| **3** | **39.8** | **233.1** | **5.85x** |
| **4** | **39.9** | **297.7** | **7.46x** |
| 8 | 104.6 | 509.4 | 4.87x |
| 16 | 205.7 | **786.0** | 3.82x |

*(aggregate tok/s, Qwen3.6-35B-A3B-NVFP4, TP2, MML 16384, MNS 16, temp 0)*

Aggregate now rises monotonically through N=16 and per-stream tapers gracefully
(95.1 -> 49.1 tok/s across a 16x concurrency increase) instead of collapsing.

### Validated on three models and two TP configurations (2026-08-27)

The fix is a property of the stack, not of one checkpoint. Two arms per model, differing
**only** in `cudagraph_capture_sizes`; greedy (temp 0):

| model | size / TP | N=4 stock `[1,2]` | N=4 ladder | speedup |
|---|---|---:|---:|---:|
| Qwen3.6-35B-A3B-NVFP4 | 22 GB, TP2 | 39.9 | 297.7 | **7.46x** |
| Ornith-1.5-35B-A3B-NVFP4 | 22 GB, TP2 | 30.9 | 282.9 | **9.16x** |
| Laguna-S-2.1-NVFP4 | 92 GB, **TP4** | 30.1 | 79.2 | **2.63x** |

**N=1 and N=2 are the built-in control.** Those widths are captured under *both* arms, so they
must agree — and they do to within 0.3% on every model (e.g. Ornith 95.66 vs 95.36 at N=1,
171.82 vs 171.65 at N=2). Divergence appears only at widths the stock list does not cover, which
is what the dispatcher explanation predicts and what a confound would not produce.

Power draw corroborates independently: at N=4 the stock arm sits at 53.9-59.2 W mean while the
ladder arm draws 84.7-88.4 W — the difference between launching kernels eagerly and replaying a
captured graph.

Magnitude varies with how GEMM-bound the model is (Laguna is ~3x slower per stream, so
graph-launch overhead is a smaller share of its step time), but **the direction and mechanism are
identical everywhere**.

### What was actually wrong

1Cat-vLLM pins the SM70 cudagraph capture list to `[1, 2]`
(`vllm/config/vllm.py`, the `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH` branch). The
dispatcher (`vllm/v1/cudagraph_dispatcher.py`) then returns `CUDAGraphMode.NONE` for any
step wider than the largest captured size:

```python
if ... or num_tokens > max_size:      # 3 > 2
    return CUDAGraphMode.NONE, ...    # fully eager, no piecewise fallback
```

A decode step is `N*(K+1)` tokens wide, so at K=0 **every step with 3+ running requests
ran fully eager** — thousands of tiny kernel launches, CPU-bound. Three independent
confirmations: forcing `--enforce-eager` at N=2 reproduces the collapse (9.2 tok/s per
stream) at a concurrency that otherwise runs fine; power draw sits at 48-55 W when eager
versus 64-83 W with graphs (the GPU was starved, not saturated); and the route census is
identical in both arms, so the kernels were never the variable.

This also corrects an earlier claim in this README: the collapse was called
"a property of the shared serving path" and "TP-independent". TP-independent was right,
but for the wrong reason — every TP hit it because the capture list is TP-independent.

### What you need to do

Nothing, if you use `serve.sh` — it now builds a capture ladder from `MNS` and `K`
automatically and prints it at launch. To override or opt out:

```bash
CAPTURE=1,2,4,8,16 MNS=16 ./serve.sh    # explicit ladder
CAPTURE=none       MNS=16 ./serve.sh    # stock behaviour (reproduces the cliff)
```

Launching vLLM directly, pass the ladder yourself — sizes must cover `N*(K+1)` for every
N the scheduler can schedule:

```bash
--compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,6,8,12,16]}'   # K=0, MNS=16
```

⚠️ `VLLM_SM70_DENSE_CUDAGRAPH_CAPTURE=1` looks like the fix and is a **no-op** — its
branch is unreachable once the `[1,2]` assignment has run.

⚠️ Capture costs startup time: PIECEWISE capture is ~35 s per size against 1.6-2.5 s for
FULL decode capture. `VLLM_SM70_FLASH_V100_0DOT3_DECODE_ONLY_CAPTURE=1` skips the
expensive half if boot time matters more than mixed prefill-decode graphs.

### Multi-instance layouts (still valid, now less necessary)

- **Split the array.** Two engines on *disjoint* GPUs at N=2 each interfere by <2%. Three
  TP2 instances across six V100s reached **187.1 tok/s** vs 77.3 for one TP4 on four cards.
- **NVIDIA MPS, if engines must share cards.** Two engines at N=2 each under MPS:
  **259.7 tok/s at 4 total streams** vs 35.7 for one engine; without MPS the same config
  collapses to 38.6. Start `nvidia-cuda-mps-control -d` **before** the serving processes —
  it cannot attach to running CUDA contexts. Sharing costs ~23% per instance.

Those numbers were measured under the old capture list, so they understate what a single
engine can now do. A single TP2 engine at N=16 reaches 786 tok/s.

---

## Install

```bash
git clone https://github.com/gilby/volta-nvfp4 && cd volta-nvfp4
./scripts/bootstrap-sm70.sh
```

`bootstrap-sm70.sh` installs the pinned 1Cat-vLLM 1.2.2 wheel (SHA256-verified), deploys the
[`fork_patches/`](fork_patches/README.md) over the installed package keeping a `.pre_bootstrap`
backup of every file it replaces, and warms the kernel JIT. Every path derives from the checkout
or an environment variable — nothing is specific to our machines.

Requires CUDA 12.8, `TORCH_CUDA_ARCH_LIST=7.0`, and Python 3.12.

## Serve

```bash
MODEL=/models/ornith-ai/Ornith-1.5-35B-A3B-NVFP4 SERVED=ornith15 \
GPU=4 TP=1 MML=131072 MNS=8 ./serve.sh
```

`serve.sh --help` is the header comment; **per-model settings, sweeps and pitfalls are in
[`MODELS.md`](MODELS.md)** — use those values, not the conservative defaults.

## Test hardware

**6× Tesla V100-PCIE-32GB** (not SXM2). **No NVLink** — all-PHB topology, every GPU↔GPU hop
crosses the host bridge; PCIe Gen3 ×16 (~13 GB/s per card); 377 GiB system RAM.

⚠️ **Every number in this repository was measured with the cards power-capped to 200 W, against
a 250 W default** (`nvidia-smi -pl 200`; `power.max_limit` is 250 W).

**Measured 2026-08-26: that cap is not binding for these workloads, and lifting it does not help.**
A/B on Qwen3.6-35B-A3B-NVFP4 (TP2, temp 0), sampling power and clocks once a second during
generation:

| N | 200 W aggTPS | 250 W aggTPS | peak draw / GPU | SM clock |
|---|---:|---:|---:|---:|
| 1 | 95.8 | 96.3 | 78–80 W | 1380 MHz |
| 2 | 161.3 | 174.6 | 80–83 W | 1380 MHz |
| 3 | 40.6 | 39.8 | 56 W | 1380 MHz |
| 8 | 106.5 | 103.8 | 58 W | 1380 MHz |

Peak draw never exceeded **83 W of the 200 W budget** and the SM clock sat at **1380 MHz, the
hardware maximum**, in both arms — so there is no headroom the cap was withholding. Raising to
250 W changed nothing outside noise (+0.6% at N=1, −2% at N≥3). If you are comparing your own
results against this repo, **a stock 250 W box should not expect a power-related advantage**;
check what your cards actually draw before assuming otherwise:

```bash
nvidia-smi --query-gpu=index,power.draw,power.limit,clocks.sm --format=csv   # under load
```

TP all-reduce is also materially more expensive here than on an SXM2/NVLink box, so multi-GPU
numbers are a **lower bound** for better-connected Volta systems.

---

## What this fork adds

| file | purpose |
|---|---|
| `kernels/qpn2_moe_kernel.cu` | MoE expert-routing kernel — the core addition |
| `kernels/skinny_kernels.cu` | modifications to the upstream kernel |
| `fork_patches/skinny_moe.py` | vLLM patch wiring the MoE path (`VLLM_SKINNY_MOE=1`) |
| `fork_patches/modelopt.py` | ModelOpt path, incl. a TP>1 `supports_internal_mk` fix |
| `serve.sh` | parameterized launcher for every model in `MODELS.md` |

The kernel's measured contribution is **+11% at N=1 and +21% at N=2** — entirely inside the
usable band. ≥95% of concurrent round time is outside the NVFP4 GEMM, so no amount of
GEMM-kernel work will fix concurrency here; the remaining target is GDN / linear attention.

---

## Gotchas

⚠️ **An MPS daemon scoped to a device subset makes the other GPUs vanish.** Starting it with
`CUDA_VISIBLE_DEVICES=0,1,2,3` made a later engine on GPUs 4–5 die with `RuntimeError: No CUDA
GPUs are available` while both cards sat idle at 0 MiB. It reads as a hardware fault; it is a
scoping bug.

⚠️ **`pgrep` cannot tell you whether MPS is running, over ssh.** `pgrep -x
nvidia-cuda-mps-control` never matches (24-char name vs the 15-char `comm` field), and
`pgrep -f` *self-matches* on the ssh argv. The only trustworthy probe is
`echo get_server_list | nvidia-cuda-mps-control`.

⚠️ **Load co-resident engines strictly sequentially.** vLLM sizes its KV pool from *free*
memory at profiling time, so a second instance loading concurrently makes the first abort with
`ValueError: No available memory for the cache blocks` — which looks like the utilization is
set too low when it was fine. Wait for `/v1/models` before starting the next.

⚠️ **Check the GPU clock before investigating any throughput gap.** `utilization.gpu`, P-state
and throttle reasons all read normal on a clock-starved GPU.

---

## License

MIT (this repository's own code) — see [`LICENSE`](LICENSE).

The files in [`fork_patches/`](fork_patches/) are derivative works of
[vLLM](https://github.com/vllm-project/vllm), Apache-2.0, and each carries its modification
notice as Apache-2.0 §4(b) requires. See [`LICENSE-APACHE-2.0`](LICENSE-APACHE-2.0) and
[`NOTICE`](NOTICE).

## Citation

If you use this work, cite **v100-skinny** first — see [`CITATION.cff`](CITATION.cff).
