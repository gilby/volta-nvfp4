# Fork patches (1Cat-vLLM 1.2.2)

**These files are derivative works of [vLLM](https://github.com/vllm-project/vllm),
Copyright contributors to the vLLM project, licensed under Apache-2.0** — not
under this repository's MIT default. Each states its modifications in its
header as Apache-2.0 §4(b) requires. See [`../LICENSE-APACHE-2.0`](../LICENSE-APACHE-2.0)
and [`../NOTICE`](../NOTICE).

They are complete upstream sources carrying local edits, tracked here so the
changes are diffable and reversible outside the installed package. The engine
is installed as a wheel (no source checkout), so the patched files are copied
over the corresponding paths under the environment's `site-packages/`.

**Install them with [`../scripts/bootstrap-sm70.sh`](../scripts/bootstrap-sm70.sh)**,
which resolves the target paths, keeps a `.pre_bootstrap` backup of every file
it replaces, and verifies the kernel extension afterwards. Never hand-edit an
installed file: the tracked copy here is the reviewable source of truth.

| File | Install path (under `site-packages/`) | What we changed |
|---|---|---|
| `marlin.py` | `vllm/model_executor/kernels/linear/nvfp4/marlin.py` | The skinny NVFP4 dispatch: QPN2 geometry winners own decode M 1–8 including `lm_head`, per-shape (split, nacc) table, shape-aware route map; legacy QPN for M 9–16; Marlin above that. |
| `modelopt.py` | `vllm/model_executor/layers/quantization/modelopt.py` | The QPN8 FP8 W8A16 path (`mma.sync.m8n8k4`, incl. the MT=2 two-tile variant), lowers the ModelOpt minimum compute capability from SM89 to SM70, adds route/census logging. **This is what lets a published mixed FP8+NVFP4 checkpoint load on Volta at all.** |
| `torch_utils.py` | `vllm/utils/torch_utils.py` | KV-dtype policy: a checkpoint's `kv_cache_quant_algo` describes how its *weights* were made and is no longer honoured as a KV-cache directive below SM80. Without this the verbatim checkpoint silently booted an FP8 KV cache and lost the tensor-core decode route (+4.82 ms/round). |
| `attention.py` | `vllm/model_executor/layers/attention/attention.py` | The same policy on the compressed-tensors re-apply path. |
| `gpu_model_runner.py` | `vllm/v1/worker/gpu_model_runner.py` | Persistent-metadata speculative round, a per-phase GPU profiler, and NVTX phase brackets for per-kernel attribution. |
| `gdn_attn.py` | `vllm/v1/attention/backends/gdn_attn.py` | Chain-MTP GDN fast metadata build (−1.4 ms/step, byte-identical output). |
| `custom_all_reduce.py` | `vllm/distributed/device_communicators/custom_all_reduce.py` | All-reduce residency instrumentation, default off. Measurement tool, dormant in production. |
