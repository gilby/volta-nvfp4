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

## ⚠️ Read this before deploying: the `N ≤ 2` rule

**Throughput peaks at N=2 concurrent streams and collapses at N=3** — a third simultaneous
user costs 71–86% of aggregate throughput.

This is **not** the MoE kernel. With the kernel disabled (`MOEFLAG=0`, stock Marlin) the
collapse is identical, and per-stream throughput at N=3 lands within 4% of the kernel-on arm
(11.5 vs 11.1 tok/s). It is a property of the shared serving path on this SM70 stack, and it
is **TP-independent** — TP1, TP2 and TP4 all break at exactly N=3, despite TP4 bringing 4× the
weight-streaming bandwidth and 6.1× the KV pool.

**Treat this as a single-user / small-team stack, not a shared serving endpoint.**

Two ways around it, both measured:

- **Split the array.** Two engines on *disjoint* GPUs at N=2 each interfere by <2%. Three TP2
  instances across six V100s reached **187.1 tok/s** vs 77.3 for one TP4 on four cards.
- **NVIDIA MPS, if engines must share cards.** Two engines at N=2 each under MPS: **259.7 tok/s
  at 4 total streams** vs 35.7 for one engine. Without MPS the same config collapses to 38.6.
  Start `nvidia-cuda-mps-control -d` **before** the serving processes — it cannot attach to
  running CUDA contexts. Sharing still costs ~23% per instance, so prefer splitting.

⚠️ This breakpoint is invisible to a standard N=1/2/4/8 sweep, which steps straight over N=3.
It was only found by measuring N=3 explicitly.

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

TP all-reduce is materially more expensive here than on an SXM2/NVLink box, so multi-GPU
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
