#!/usr/bin/env bash
# Serve an NVFP4 MoE model on Tesla V100 (SM70) through the QPN2-MoE kernels.
#
# Replaces the previous per-model serve_*.sh scripts. Per-model values for
# MODEL / MML / MNS / TP / K / GENCFG are in MODELS.md — use those, not the
# defaults here, which are deliberately conservative.
#
#   MODEL     required. Path to the checkpoint.
#   SERVED    required. Served model name.
#   GPU       CUDA_VISIBLE_DEVICES, e.g. 4 or 0,1 (default 0)
#   TP        tensor-parallel size (default 1; must match the GPU count)
#   MML       max-model-len (default 16384)
#   MNS       max-num-seqs (default 1 — see the N<=2 rule in README.md)
#   K         MTP speculative depth, 0 = off (default 0)
#   PORT      default 8033
#   GMU       gpu-memory-utilization (default 0.92)
#   MOEFLAG   VLLM_SKINNY_MOE, 1 = QPN2-MoE on (default 1; 0 = stock Marlin A/B arm)
#   LMHEAD    VLLM_SKINNY_LMHEAD (default 1; set 0 for BF16-lm_head checkpoints)
#   KVDTYPE   optional --kv-cache-dtype (e.g. fp8_e4m3)
#   GENCFG    optional --override-generation-config JSON
#   ATTN      attention backend (default FLASH_ATTN_V100; Gemma-4 needs TRITON_ATTN)
#   REPO/MOE  checkout locations (default ~/v100-skinny and this directory)
set -u
REPO=${REPO:-~/v100-skinny}
MOE=${MOE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
PY=$REPO/.venv-sm70/bin/python
K=${K:-0}; PORT=${PORT:-8033}; MOEFLAG=${MOEFLAG:-1}; LMHEAD=${LMHEAD:-1}
LOG=${LOG:-$MOE/serve.log}
rm -f "$LOG"

SPEC=()
if [ "$K" -gt 0 ]; then
  K1=$((K+1)); K2=$((K1*2))
  SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$K,\"draft_sample_method\":\"greedy\",\"use_local_argmax_reduction\":true}"
        --compilation-config "{\"cudagraph_capture_sizes\":[$K1,$K2]}")
fi

PATH=$REPO/.venv-sm70/bin:$PATH \
CUDA_VISIBLE_DEVICES=${GPU:-0} \
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8} \
TORCH_CUDA_ARCH_LIST=7.0 \
VLLM_SM70_NVFP4_TURBOMIND=0 \
VLLM_SM70_QUANT_BACKEND=marlin \
VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1 \
VLLM_SKINNY_NVFP4=1 \
VLLM_SKINNY_QPN=1 \
VLLM_SKINNY_QPN2=1 \
VLLM_SKINNY_LMHEAD=$LMHEAD \
VLLM_SKINNY_DROP_CT=1 \
VLLM_SKINNY_MOE=$MOEFLAG \
VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0 \
VLLM_SKINNY_NVFP4_SRC=$MOE/kernels/skinny_kernels.cu \
VLLM_SM70_QPN8_MT2=1 \
VLLM_FLASH_V100_DECODE_PARTITION_SIZE=256 \
setsid $PY -m vllm.entrypoints.openai.api_server \
  --model "${MODEL:?set MODEL}" \
  --served-model-name "${SERVED:?set SERVED}" \
  ${GENCFG:+--override-generation-config "$GENCFG"} \
  --trust-remote-code \
  --dtype float16 \
  --attention-backend "${ATTN:-FLASH_ATTN_V100}" \
  --tensor-parallel-size "${TP:-1}" \
  --gpu-memory-utilization "${GMU:-0.92}" \
  --max-model-len "${MML:-16384}" \
  --max-num-seqs "${MNS:-1}" \
  --max-num-batched-tokens 4096 \
  ${KVDTYPE:+--kv-cache-dtype "$KVDTYPE"} \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  "${SPEC[@]}" \
  --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 < /dev/null &
echo $! > "$MOE/serve.pid"
echo "launched pid $(cat "$MOE/serve.pid") TP=${TP:-1} port=$PORT K=$K MOE=$MOEFLAG LMHEAD=$LMHEAD"
