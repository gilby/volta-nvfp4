// ---------------------------------------------------------------------------
// QPN2-MoE: grouped / expert-indirected QPN2 for routed experts (SM70).
//
// Same dataflow as skinny_nvfp4_qpn2 — quadpairs on N, one 32-column tile per
// blockIdx.x, split-K over warps, m8n8k4 with FP16 A/B and FP32 accumulate,
// gscale folded into the group scales in-kernel (the FP16-outlier fix) — with
// three additions for MoE:
//
//   1. blockIdx.y walks moe_align_block_size()'s row-blocks: up to 8 slots
//      (token, expert) pairs sharing ONE expert. Weight pointers are offset
//      by that expert's prepacked block. All eight MMA rows stay inside one
//      expert, so the inner loop is byte-identical to the dense kernel.
//   2. Row indirection: A-fragment rows come from sorted_slots[b*8+r]; for
//      the first GEMM a slot's activation row is slot/topk (its token); for
//      the second it is the slot itself (in_div == 1). Output rows are the
//      slot ids, so y is written exactly once per real slot, no atomics.
//   3. gscales is per (expert, tile): w13 concatenates gate and up along N,
//      whose weight_scale_2 may differ; 512 % 32 == 0 keeps every tile on
//      one side of the seam, so a per-tile gscale resolves it exactly.
//
// Pad slots (sorted id >= S) load zero A-rows and skip the store. Blocks at
// or beyond *ntpp (num_tokens_post_padded) exit before touching weights.
template <int SPLITK, int NACC>
__global__ void skinny_nvfp4_qpn2_moe(
    const uint8_t *__restrict__ bcodes, const uint8_t *__restrict__ bscales,
    const float *__restrict__ gscales, const half *__restrict__ x,
    const int *__restrict__ block_expert, const int *__restrict__ sorted_slots,
    const int *__restrict__ ntpp, half *__restrict__ y, int N, int K, int S,
    int in_div) {
  __shared__ float cs[SPLITK > 1 ? SPLITK : 1][SPLITK > 1 ? 256 : 1];

  const int b = blockIdx.y;
  if (b * 8 >= *ntpp) return;
  const int e = block_expert[b];
  if (e < 0) return;

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int TN = N >> 5;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int G = K >> 4, Gq = G / SPLITK;
  const int g0 = warp * Gq;
  const size_t estride = (size_t)TN * G * 32;  // uint2 units == scale bytes
  const uint2 *cb = reinterpret_cast<const uint2 *>(bcodes) +
                    (size_t)e * estride + (size_t)tile * G * 32 + lane;
  const uint8_t *sb =
      bscales + (size_t)e * estride + (size_t)tile * G * 32 + lane;

  const float gscale = gscales[(size_t)e * TN + tile];
  const half2 gm2 = __float2half2_rn(gscale * 16384.f);

  const int in_slot = sorted_slots[b * 8 + r];
  const bool rvalid = in_slot < S;
  const half *xrow = x + (size_t)(rvalid ? in_slot / in_div : 0) * K;

  float c[NACC][8];
#pragma unroll
  for (int a = 0; a < NACC; a++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[a][i] = 0.f;

#pragma unroll 4
  for (int g = g0; g < g0 + Gq; g++) {
    const uint2 q2 = __ldcs(cb + (size_t)g * 32);
    const half2 sc2 =
        __hmul2(fp8e4m3_to_half2(__ldg(sb + (size_t)g * 32)), gm2);
    half2 bfrag[8];
    dequant8_tm(q2.x, sc2, bfrag + 0);
    dequant8_tm(q2.y, sc2, bfrag + 4);
    const unsigned *B = reinterpret_cast<const unsigned *>(bfrag);
    uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
    if (rvalid) {
      a01 = *reinterpret_cast<const uint4 *>(xrow + g * 16);
      a23 = *reinterpret_cast<const uint4 *>(xrow + g * 16 + 8);
    }
    const unsigned *A0 = reinterpret_cast<const unsigned *>(&a01);
    const unsigned *A1 = reinterpret_cast<const unsigned *>(&a23);
    MMA_8N8K4(c[0], A0[0], A0[1], B[0], B[1]);
    MMA_8N8K4(c[1 % NACC], A0[2], A0[3], B[2], B[3]);
    MMA_8N8K4(c[2 % NACC], A1[0], A1[1], B[4], B[5]);
    MMA_8N8K4(c[3 % NACC], A1[2], A1[3], B[6], B[7]);
  }

#pragma unroll
  for (int a = 1; a < NACC; a++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[0][i] += c[a][i];

  if (SPLITK == 1) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      const int oslot = sorted_slots[b * 8 + row];
      if (oslot < S)
        y[(size_t)oslot * N + (size_t)tile * 32 + qp * 8 + col] =
            __float2half(c[0][i]);
    }
    return;
  }

#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    cs[warp][row * 32 + qp * 8 + col] = c[0][i];
  }
  __syncthreads();
  for (int e2 = threadIdx.x; e2 < 256; e2 += blockDim.x) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < SPLITK; w++) v += cs[w][e2];
    const int row = e2 >> 5, col = e2 & 31;
    const int oslot = sorted_slots[b * 8 + row];
    if (oslot < S)
      y[(size_t)oslot * N + (size_t)tile * 32 + col] = __float2half(v);
  }
}

torch::Tensor skinny_gemm_qpn2_moe(torch::Tensor x, torch::Tensor qcodes,
                                   torch::Tensor qscales,
                                   torch::Tensor gscales,
                                   torch::Tensor block_expert,
                                   torch::Tensor sorted_slots,
                                   torch::Tensor ntpp, int64_t s_out,
                                   int64_t n, int64_t k, int64_t in_div,
                                   int64_t splitk, int64_t nacc) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(qcodes.is_cuda() && qcodes.dtype() == torch::kUInt8 &&
              qcodes.is_contiguous());
  TORCH_CHECK(qscales.is_cuda() && qscales.dtype() == torch::kUInt8 &&
              qscales.is_contiguous());
  TORCH_CHECK(gscales.is_cuda() && gscales.dtype() == torch::kFloat &&
              gscales.is_contiguous());
  TORCH_CHECK(block_expert.is_cuda() && block_expert.dtype() == torch::kInt &&
              block_expert.is_contiguous());
  TORCH_CHECK(sorted_slots.is_cuda() && sorted_slots.dtype() == torch::kInt &&
              sorted_slots.is_contiguous());
  TORCH_CHECK(ntpp.is_cuda() && ntpp.dtype() == torch::kInt);
  TORCH_CHECK(k % 64 == 0 && (k / 16) % splitk == 0, "K/SPLITK");
  TORCH_CHECK(n % 32 == 0, "N % 32");
  TORCH_CHECK(x.size(1) == k, "x K mismatch");
  TORCH_CHECK(sorted_slots.numel() == block_expert.numel() * 8,
              "sorted_slots must be blocks*8");
  const int64_t nblocks = block_expert.numel();
  auto y = torch::empty({s_out, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_QPN2_MOE(SPv, NAv)                                            \
  skinny_nvfp4_qpn2_moe<SPv, NAv>                                            \
      <<<dim3((int)(n / 32), (int)nblocks), dim3(32 * SPv), 0, stream>>>(    \
          qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),           \
          gscales.data_ptr<float>(),                                         \
          reinterpret_cast<const half *>(x.data_ptr<at::Half>()),            \
          block_expert.data_ptr<int>(), sorted_slots.data_ptr<int>(),        \
          ntpp.data_ptr<int>(),                                              \
          reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,  \
          (int)s_out, (int)in_div)

  const int key = (int)(splitk * 10 + nacc);
  switch (key) {
    case 21: LAUNCH_QPN2_MOE(2, 1); break;
    case 41: LAUNCH_QPN2_MOE(4, 1); break;
    case 81: LAUNCH_QPN2_MOE(8, 1); break;
    case 82: LAUNCH_QPN2_MOE(8, 2); break;
    case 161: LAUNCH_QPN2_MOE(16, 1); break;
    case 162: LAUNCH_QPN2_MOE(16, 2); break;
    default: TORCH_CHECK(false, "qpn2_moe splitk in {2,4,8,16}, nacc {1,2}");
  }
#undef LAUNCH_QPN2_MOE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
