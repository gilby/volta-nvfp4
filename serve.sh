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
#   MNS       max-num-seqs (default 1). Concurrency is NOT capped at 2 any more —
#             the capture ladder below is what made N>2 usable; see README.md
#   CAPTURE   cudagraph capture sizes, csv. Default: derived from MNS and K.
#             "none" omits the flag (stock behaviour = the N>2 cliff)
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

MNS=${MNS:-1}

# --- cudagraph capture ladder (THE concurrency lever; see README) -------------
# A decode step is N*(K+1) tokens wide for N running requests. vLLM dispatches
# CUDAGraphMode.NONE - fully eager - for any width above the largest captured
# size, and eager decode on SM70 is CPU-launch-bound: measured 5.9x slower at
# N=3 and 7.5x at N=4. So capture EVERY width the scheduler can produce, not a
# token pair. CAPTURE=<csv> overrides; CAPTURE=none omits the flag entirely.
build_ladder() {                      # $1=MNS  $2=K
  local mns=$1 k=$2 w=$(( $2 + 1 )) n step out=""
  step=1; [ "$mns" -gt 16 ] && step=$(( (mns + 15) / 16 ))   # <=16 entries
  for (( n=1; n<=mns; n+=step )); do out="${out}${out:+,}$(( n * w ))"; done
  [ $(( (mns - 1) % step )) -ne 0 ] && out="${out},$(( mns * w ))"
  echo "$out"
}
if [ "${CAPTURE:-}" = "none" ]; then
  CAP=()
else
  LADDER=${CAPTURE:-$(build_ladder "$MNS" "$K")}
  CAP=(--compilation-config "{\"cudagraph_capture_sizes\":[$LADDER]}")
fi

SPEC=()
if [ "$K" -gt 0 ]; then
  SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$K,\"draft_sample_method\":\"greedy\",\"use_local_argmax_reduction\":true}")
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
  --max-num-seqs "$MNS" \
  --max-num-batched-tokens 4096 \
  ${KVDTYPE:+--kv-cache-dtype "$KVDTYPE"} \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  "${SPEC[@]}" "${CAP[@]}" \
  --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 < /dev/null &
echo $! > "$MOE/serve.pid"
echo "launched pid $(cat "$MOE/serve.pid") TP=${TP:-1} port=$PORT K=$K MNS=$MNS MOE=$MOEFLAG LMHEAD=$LMHEAD capture=[${LADDER:-none}]"

# ---------------------------------------------------------------------------
# Readiness gate + JIT warm-up.  WARMUP=0 to skip.
#
# Without this, serve.sh returns as soon as the process is forked and the FIRST REAL REQUEST
# pays kernel JIT + CUDA-graph capture. That penalty is large and it lands on a user:
# measured on this box, an identical model loaded in 350 s cold vs 150 s with a warm page
# cache, and the mt2 work saw a first-request decode penalty down at ~28 tok/s.
# It matters most under llama-swap, where a swap-in is triggered BY a user request.
#
# NOTE: send ONE blocking request and wait. Do not poll a swap-managed endpoint during a cold
# start - each probe re-triggers the swap and can kill the in-flight load.
if [ "${WARMUP:-1}" = "1" ]; then
  t0=$(date +%s)
  ready=0
  for i in $(seq 1 "${WARMUP_TRIES:-240}"); do
    if curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED"; then
      ready=1; break
    fi
    # surface a dead load instead of silently waiting out the timeout
    if ! kill -0 "$(cat "$MOE/serve.pid")" 2>/dev/null; then
      echo "warmup: server process died during load - see $LOG"; break
    fi
    sleep 5
  done
  if [ "$ready" = "1" ]; then
    echo "warmup: endpoint up after $(( $(date +%s) - t0 ))s; sending one blocking generation to force JIT + graph capture"
    w0=$(date +%s)
    body=$(printf '{"model":"%s","messages":[{"role":"user","content":"Write three sentences about heat transfer."}],"max_tokens":192,"temperature":0}' "$SERVED")
    out=$(curl -sf -m "${WARMUP_TIMEOUT:-900}" "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H 'Content-Type: application/json' -d "$body" 2>/dev/null)
    # count reasoning too: a reasoning model can spend a small budget entirely in <think>
    # and return empty content on a perfectly healthy server.
    n=$(printf '%s' "$out" | python3 -c 'import sys,json
try:
    m=json.load(sys.stdin)["choices"][0]["message"]
    print(len(((m.get("content") or "")+(m.get("reasoning_content") or "")).strip()))
except Exception:
    print(0)' 2>/dev/null)
    if [ "${n:-0}" -gt 0 ]; then
      echo "warmup: OK - generated $n chars in $(( $(date +%s) - w0 ))s; kernels compiled, graphs captured"
    else
      echo "warmup: FAILED - endpoint answered but produced nothing (see $LOG)"
    fi
  else
    echo "warmup: endpoint never came up within $(( ${WARMUP_TRIES:-240} * 5 ))s"
  fi
fi
