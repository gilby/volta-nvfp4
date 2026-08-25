# SPDX-License-Identifier: Apache-2.0
"""Skinny QPN2-MoE: expert-indirected QPN2 for W4A16_NVFP4 routed experts on
SM70 (Volta), replacing the Marlin MoE path at every batch width.

Enabled by VLLM_SKINNY_MOE=1. Design mirrors the dense v100-skinny stack:
weights stay in the checkpoint's own 4-bit codes, pre-permuted once at load
into m8n8k4 fragment order (one contiguous block per expert), decoded to FP16
registers at the point of consumption. Slot blocking comes from vLLM's own
moe_align_block_size (block_size=8 == the QPN2 MMA row width), so every
launch is one grid over (N/32 tiles) x (row blocks), CUDA-graph capturable.

Numerics: identical dequant path to dense QPN2, including the in-kernel
global-scale fold (gscale * 16384 with the /16384 baked into the decoded
magnitudes) that fixed the FP16 activation-outlier overflow. w13's gate and
up halves keep their own weight_scale_2 via a per-(expert, tile) gscale
table; 512 % 32 == 0 keeps tiles from straddling the gate/up seam.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

SKINNY_MOE_ENABLED = os.environ.get("VLLM_SKINNY_MOE", "0") == "1"

_route_counts: dict[tuple, int] = {}
_route_seen: set = set()


def _ext():
    from vllm.model_executor.kernels.linear.nvfp4.marlin import (
        _get_skinny_ext,
    )

    return _get_skinny_ext()


def _prepack_stacked(codes: torch.Tensor, scales: torch.Tensor):
    """Prepack [E, N, K/2] u8 codes + [E, N, K/16] fp8-as-u8 scales for all
    experts in one _qpn_prepack call: tiles never straddle expert boundaries
    (N % 32 == 0), so viewing the stack as one [E*N, K/2] matrix yields an
    expert-major fragment-order layout directly."""
    from vllm.model_executor.kernels.linear.nvfp4.marlin import _qpn_prepack

    e, n, k2 = codes.shape
    qc, qs = _qpn_prepack(
        codes.reshape(e * n, k2).contiguous(),
        scales.reshape(e * n, scales.shape[-1]).contiguous(),
    )
    assert qc is not None, "expert shape not tile-aligned"
    return qc.contiguous(), qs.contiguous()


def _pick_splitk(k: int, n_tiles_total: int) -> int:
    # K/16 groups must divide by splitk; prefer deeper splits for skinny
    # grids, shallower when the slot dimension already fills the machine.
    g = k // 16
    for s in (8, 4, 2):
        if g % s == 0:
            return s
    return 1


def prepack_experts(layer) -> None:
    """Replace Marlin repack: build QPN fragment-order expert blocks and a
    per-(expert, tile) gscale table; free the checkpoint staging tensors."""
    w13 = layer.w13_weight.data  # [E, N13, K/2] u8
    w13_s = layer.w13_weight_scale.data  # [E, N13, K/16] fp8
    w13_s2 = layer.w13_weight_scale_2.data  # [E, 2] f32
    w2 = layer.w2_weight.data  # [E, N2, K2/2] u8
    w2_s = layer.w2_weight_scale.data
    w2_s2 = layer.w2_weight_scale_2.data  # [E] f32 (or [E,1])

    e, n13, k13_2 = w13.shape
    _, n2, k2_2 = w2.shape
    assert w13.dtype == torch.uint8 and w2.dtype == torch.uint8

    qc13, qs13 = _prepack_stacked(w13, w13_s.view(torch.uint8))
    qc2, qs2 = _prepack_stacked(w2, w2_s.view(torch.uint8))

    tn13 = n13 // 32
    tn2 = n2 // 32
    # gate rows are [0, n13/2), up rows [n13/2, n13): per-tile gscale.
    gs13 = torch.empty(e, tn13, dtype=torch.float32, device=w13.device)
    gs13[:, : tn13 // 2] = w13_s2[:, 0:1].float()
    gs13[:, tn13 // 2 :] = w13_s2[:, 1:2].float()
    gs2 = (
        w2_s2.reshape(e, 1).float().expand(e, tn2).contiguous()
    )

    layer.skinny_moe_qc13 = qc13
    layer.skinny_moe_qs13 = qs13
    layer.skinny_moe_gs13 = gs13.contiguous()
    layer.skinny_moe_qc2 = qc2
    layer.skinny_moe_qs2 = qs2
    layer.skinny_moe_gs2 = gs2
    layer.skinny_moe_shape = (e, n13, k13_2 * 2, n2, k2_2 * 2)
    layer.skinny_moe_splitk13 = _pick_splitk(k13_2 * 2, e * tn13)
    layer.skinny_moe_splitk2 = _pick_splitk(k2_2 * 2, e * tn2)

    logger.info(
        "Skinny MoE prepack: E=%d w13[N=%d K=%d] w2[N=%d K=%d] "
        "splitk=(%d,%d) bytes=%.2f GiB",
        e, n13, k13_2 * 2, n2, k2_2 * 2,
        layer.skinny_moe_splitk13, layer.skinny_moe_splitk2,
        (qc13.numel() + qs13.numel() + qc2.numel() + qs2.numel()) / 2**30,
    )


def moe_apply(
    layer,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts,
    shared_experts_input,
) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExpertsOrder,
    )

    if shared_experts is not None:
        # Conditional no-op unless this config resolves to MK-internal
        # ordering; mirrors what the modular kernels do.
        shared_experts.apply(
            shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
        )

    e, n13, k13, n2, k2 = layer.skinny_moe_shape
    t, k = x.shape
    topk = topk_ids.shape[1]
    s = t * topk
    assert k == k13, f"hidden {k} != w13 K {k13}"
    assert not getattr(layer, "apply_router_weight_on_input", False), (
        "skinny MoE applies router weights on output"
    )
    assert getattr(layer, "expert_map", None) is None, (
        "skinny MoE supports TP1/no-EP only (expert_map must be None)"
    )

    sorted_ids, expert_blk, ntpp = moe_align_block_size(topk_ids, 8, e)
    sorted_ids = sorted_ids.to(torch.int32)
    expert_blk = expert_blk.to(torch.int32)
    ntpp = ntpp.to(torch.int32)

    ext = _ext()
    xh = x if x.dtype == torch.float16 else x.half()
    h1 = ext.gemm_qpn2_moe(
        xh.contiguous(), layer.skinny_moe_qc13, layer.skinny_moe_qs13,
        layer.skinny_moe_gs13, expert_blk, sorted_ids, ntpp,
        s, n13, k13, topk, layer.skinny_moe_splitk13, 1,
    )  # [S, N13]
    i_half = n13 // 2
    act = torch.nn.functional.silu(h1[:, :i_half]) * h1[:, i_half:]
    h2 = ext.gemm_qpn2_moe(
        act.contiguous(), layer.skinny_moe_qc2, layer.skinny_moe_qs2,
        layer.skinny_moe_gs2, expert_blk, sorted_ids, ntpp,
        s, n2, k2, 1, layer.skinny_moe_splitk2, 1,
    )  # [S, N2]

    key = ("qpn2_moe", t)
    if key not in _route_seen:
        _route_seen.add(key)
        logger.info(
            "Skinny MoE route: T=%d topk=%d S=%d blocks=%d -> qpn2_moe",
            t, topk, s, expert_blk.numel(),
        )

    out = (
        h2.view(t, topk, n2).float() * topk_weights.view(t, topk, 1).float()
    ).sum(dim=1)
    return out.to(x.dtype)
