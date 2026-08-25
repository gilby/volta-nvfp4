# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Modified by the v100-skinny contributors, 2026, from 1Cat-vLLM 1.2.2
# (https://github.com/1CatAI/1Cat-vLLM). Licensed under Apache-2.0.
# Changes: adds env-gated all-reduce residency instrumentation
# (VLLM_SM70_AR_EVT, default off) -- CUDA event pairs baked into captured
# graphs plus a harvester thread. Measurement tool, dormant in production.

from contextlib import contextmanager
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.all_reduce_utils import (
    CUSTOM_ALL_REDUCE_MAX_SIZES,
    gpu_p2p_access_check,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

try:
    ops.meta_size()
    custom_ar = True
except Exception:
    # For CPUs
    custom_ar = False

logger = init_logger(__name__)


def _can_p2p(rank: int, world_size: int) -> bool:
    for i in range(world_size):
        if i == rank:
            continue
        if envs.VLLM_SKIP_P2P_CHECK:
            logger.debug("Skipping P2P check and trusting the driver's P2P report.")
            return torch.cuda.can_device_access_peer(rank, i)
        if not gpu_p2p_access_check(rank, i):
            return False
    return True


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )



# --- AR residency instrumentation (VLLM_SM70_AR_EVT=1) ----------------------
# Profiler-off confirmation of all-reduce kernel residency: CUDA event
# pairs are baked around the registered AR inside captured graphs (the
# same Event objects re-record on every replay), and eager ARs get their
# own pairs. A periodic harvest on the eager path prints distributions.
# Graceful: any event failure disables the instrumentation permanently.
import os as _os

_AR_EVT = _os.environ.get("VLLM_SM70_AR_EVT", "0") == "1"
_AR_EVT_OK = [True]
_AR_GRAPH_PAIRS: list = []          # baked into graphs, re-record at replay
_AR_EAGER_PAIRS: list = []          # rolling recent eager pairs
_AR_CALLS = [0]


def _ar_evt_new_pair():
    try:
        try:
            pre = torch.cuda.Event(enable_timing=True, external=True)
            post = torch.cuda.Event(enable_timing=True, external=True)
        except TypeError:
            pre = torch.cuda.Event(enable_timing=True)
            post = torch.cuda.Event(enable_timing=True)
        return pre, post
    except Exception:
        _AR_EVT_OK[0] = False
        return None, None


def _ar_evt_stats(pairs):
    vals = []
    for pre, post in pairs:
        try:
            if post.query():
                us = pre.elapsed_time(post) * 1000.0
                if 0.0 < us < 5000.0:   # filter torn reads across replays
                    vals.append(us)
        except Exception:
            continue
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    return (n, sum(vals) / n, vals[n // 2], vals[int(n * 0.95)], vals[-1])


def _ar_evt_thread():
    import threading, time as _t

    def loop():
        while True:
            _t.sleep(5.0)
            try:
                _ar_evt_harvest()
            except Exception:
                return
    th = threading.Thread(target=loop, daemon=True,
                          name="ar-evt-harvester")
    th.start()


_AR_THREAD = [False]


_AR_LAST_VALS: dict = {}            # pair index -> us at previous harvest


def _ar_evt_harvest():
    # Indexed per-op probe. Events re-record on every replay of their
    # graph, so a pair whose value is unchanged since the last harvest
    # belongs to a graph that did not replay (e.g. the idle capture-size
    # graph) — group-level liveness separates live decode graphs from
    # stale ones. Within a group, append order is execution order, so
    # the ordinal names the op's position in that graph.
    groups: dict = {}
    for idx, (pre, post, shape) in enumerate(_AR_GRAPH_PAIRS):
        try:
            if not post.query():
                continue
            us = pre.elapsed_time(post) * 1000.0
        except Exception:
            continue
        if not 0.0 < us < 5000.0:       # torn read across replays
            continue
        changed = _AR_LAST_VALS.get(idx) != us
        _AR_LAST_VALS[idx] = us
        groups.setdefault(shape, []).append((us, changed))
    for shape, vals in sorted(groups.items()):
        tag = "x".join(str(d) for d in shape)
        n_changed = sum(1 for _, c in vals if c)
        if n_changed < max(1, len(vals) // 10):
            print(f"[ar-evt] g{tag}: n={len(vals)} STALE", flush=True)
            continue
        us_sorted = sorted(u for u, _ in vals)
        n = len(us_sorted)
        bursts = sorted(
            ((u, i) for i, (u, _) in enumerate(vals) if u > 100.0),
            reverse=True)[:20]
        btxt = " ".join(f"{i}:{u:.0f}" for u, i in bursts)
        print(f"[ar-evt] g{tag}: n={n} changed={n_changed} "
              f"med={us_sorted[n // 2]:.1f}us p95={us_sorted[int(n * 0.95)]:.1f}us "
              f"max={us_sorted[-1]:.1f}us bursts[op:us]={btxt}", flush=True)
    e = _ar_evt_stats(_AR_EAGER_PAIRS)
    if e:
        print(f"[ar-evt] EAGER pairs n={e[0]} mean={e[1]:.1f}us "
              f"median={e[2]:.1f}us p95={e[3]:.1f}us max={e[4]:.1f}us",
              flush=True)
    del _AR_EAGER_PAIRS[:-64]
# ---------------------------------------------------------------------------


class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    # max_size: max supported allreduce size
    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size=8192 * 1024,
        symm_mem_enabled=False,
    ) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bound to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True

        if not custom_ar:
            # disable because of missing custom allreduce library
            # e.g. in a non-GPU environment
            logger.info(
                "Custom allreduce is disabled because "
                "of missing custom allreduce library"
            )
            return

        self.group = group

        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "CustomAllreduce should be attached to a non-NCCL group."
        )

        if not all(in_the_same_node_as(group, source_rank=0)):
            # No need to initialize custom allreduce for multi-node case.
            logger.warning(
                "Custom allreduce is disabled because this process group"
                " spans across nodes."
            )
            return

        rank = dist.get_rank(group=self.group)
        self.rank = rank
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size,
                str(CustomAllreduce._SUPPORTED_WORLD_SIZES),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        device_capability = current_platform.get_device_capability()
        if (
            current_platform.is_cuda()
            and symm_mem_enabled
            and device_capability is not None
        ):
            device_capability_str = device_capability.as_version_str()
            if device_capability_str in CUSTOM_ALL_REDUCE_MAX_SIZES:
                max_size = min(
                    CUSTOM_ALL_REDUCE_MAX_SIZES[device_capability_str][world_size],
                    max_size,
                )
        cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            device_ids = list(range(current_platform.device_count()))

        physical_device_id = device_ids[device.index]
        tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
        gather_list = [
            torch.tensor([0], dtype=torch.int, device="cpu") for _ in range(world_size)
        ]
        dist.all_gather(gather_list, tensor, group=self.group)
        physical_device_ids = [t.item() for t in gather_list]

        # test nvlink first, this will filter out most of the cases
        # where custom allreduce is not supported
        # this checks hardware and driver support for NVLink
        assert current_platform.is_cuda_alike()
        fully_connected = current_platform.is_fully_connected(physical_device_ids)
        if world_size > 2 and not fully_connected:
            logger.warning(
                "Custom allreduce is disabled because it's not supported on"
                " more than two PCIe-only GPUs. To silence this warning, "
                "specify disable_custom_all_reduce=True explicitly."
            )
            return
        # test P2P capability, this checks software/cudaruntime support
        # this is expensive to compute at the first time
        # then we cache the result
        # On AMD GPU, p2p is always enabled between XGMI connected GPUs
        if not current_platform.is_rocm() and not _can_p2p(rank, world_size):
            logger.warning(
                "Custom allreduce is disabled because your platform lacks "
                "GPU P2P capability or P2P test failed. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly."
            )
            return

        self.disabled = False
        # Buffers memory are owned by this Python class and passed to C++.
        # Metadata composes of two parts: metadata for synchronization and a
        # temporary buffer for storing intermediate allreduce results.
        self.meta_ptrs = self.create_shared_buffer(
            ops.meta_size() + max_size, group=group, uncached=True
        )
        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before allreduce is performed
        self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        self.fully_connected = fully_connected
        self._ptr = ops.init_custom_ar(
            self.meta_ptrs, self.rank_data, rank, self.fully_connected
        )
        ops.register_buffer(self._ptr, self.buffer_ptrs)

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def register_graph_buffers(self):
        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
        logger.info("Registering %d cuda graph addresses", len(offset))
        # We cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.
        all_data: list[list[list[int] | None]]
        all_data = [[None, None] for _ in range(dist.get_world_size(group=self.group))]
        all_data[self.rank] = [handle, offset]
        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=rank, group=self.group, device="cpu"
            )
        # Unpack list of tuples to tuple of lists.
        handles = cast(list[list[int]], [d[0] for d in all_data])
        offsets = cast(list[list[int]], [d[1] for d in all_data])
        ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.
        if self.world_size == 2 or self.fully_connected:
            return inp_size < self.max_size
        return False

    def all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor = None, registered: bool = False
    ):
        """Performs an out-of-place all reduce.

        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if out is None:
            out = torch.empty_like(inp)
        if registered:
            ops.all_reduce(self._ptr, inp, out, 0, 0)
        else:
            ops.all_reduce(
                self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size
            )
        return out

    def all_reduce_sum2(
        self,
        inp_a: torch.Tensor,
        inp_b: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(inp_a)
        ops.all_reduce_sum2(self._ptr, inp_a, inp_b, out)
        return out

    def sm70_tp2_all_reduce_gemma_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.can_sm70_tp2_all_reduce_gemma_rms_norm(
            inp, residual, weight
        ):
            raise RuntimeError("SM70 TP2 fused all-reduce RMSNorm is unavailable")
        normalized_out = torch.empty_like(inp)
        residual_out = torch.empty_like(residual, dtype=torch.float32)
        registered = self._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
        reg_buffer = 0 if registered else self.buffer_ptrs[self.rank]
        reg_buffer_size = 0 if registered else self.max_size
        ops.sm70_tp2_all_reduce_gemma_rms_norm(
            self._ptr,
            inp,
            residual,
            weight,
            normalized_out,
            residual_out,
            reg_buffer,
            reg_buffer_size,
            epsilon,
        )
        return normalized_out, residual_out

    def can_sm70_tp2_all_reduce_gemma_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> bool:
        return (
            not self.disabled
            and self.world_size == 2
            and self.should_custom_ar(inp)
            and inp.is_cuda
            and inp.dtype == torch.float16
            and inp.ndim == 2
            and 1 <= inp.shape[0] <= 64
            and inp.shape[1] == 5120
            and residual.shape == inp.shape
            and residual.dtype in (torch.float16, torch.float32)
            and weight.ndim == 1
            and weight.numel() == 5120
            and weight.dtype in (torch.float16, torch.float32)
            and inp.is_contiguous()
            and residual.is_contiguous()
            and weight.is_contiguous()
            and inp.device == residual.device == weight.device
        )

    def sm70_tp4_all_reduce_gemma_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Explicit benchmark-only TP4 path; no automatic runtime dispatch."""
        if not self.can_sm70_tp4_all_reduce_gemma_rms_norm(inp, residual, weight):
            raise RuntimeError("SM70 TP4 fused all-reduce RMSNorm is unavailable")
        normalized_out = torch.empty_like(inp)
        residual_out = torch.empty_like(residual, dtype=torch.float32)
        registered = self._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
        reg_buffer = 0 if registered else self.buffer_ptrs[self.rank]
        reg_buffer_size = 0 if registered else self.max_size
        ops.sm70_tp4_all_reduce_gemma_rms_norm(
            self._ptr,
            inp,
            residual,
            weight,
            normalized_out,
            residual_out,
            reg_buffer,
            reg_buffer_size,
            epsilon,
        )
        return normalized_out, residual_out

    def can_sm70_tp4_all_reduce_gemma_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> bool:
        return (
            not self.disabled
            and self.world_size == 4
            and self.fully_connected
            and self.should_custom_ar(inp)
            and inp.is_cuda
            and inp.dtype == torch.float16
            and inp.ndim == 2
            and 1 <= inp.shape[0] <= 64
            and inp.shape[1] == 5120
            and residual.shape == inp.shape
            and residual.dtype == torch.float32
            and weight.ndim == 1
            and weight.numel() == 5120
            and weight.dtype in (torch.float16, torch.float32)
            and inp.is_contiguous()
            and residual.is_contiguous()
            and weight.is_contiguous()
            and inp.device == residual.device == weight.device
        )

    def top1_argmax(
        self,
        input_pair: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        registered: bool = False,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty((1,), dtype=torch.int64, device=input_pair.device)
        if registered:
            ops.top1_argmax(self._ptr, input_pair, out, 0, 0)
        else:
            ops.top1_argmax(
                self._ptr,
                input_pair,
                out,
                self.buffer_ptrs[self.rank],
                self.max_size,
            )
        return out

    def tile_runtime_all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        tile_numel: int = 512,
        engine_blocks: int = 0,
        compute_iters: int = 0,
    ) -> torch.Tensor:
        if self.disabled:
            raise RuntimeError("custom allreduce is disabled")
        if out is None:
            out = torch.empty_like(inp)
        ops.tile_runtime_all_reduce(
            self._ptr,
            inp,
            out,
            self.buffer_ptrs[self.rank],
            self.max_size,
            tile_numel,
            engine_blocks,
            compute_iters,
        )
        return out

    def tile_runtime_all_reduce_engine(
        self,
        inp: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        tile_numel: int = 512,
        producer_blocks: int = 0,
        reducer_blocks: int = 0,
        compute_iters: int = 0,
    ) -> torch.Tensor:
        if self.disabled:
            raise RuntimeError("custom allreduce is disabled")
        if out is None:
            out = torch.empty_like(inp)
        ops.tile_runtime_all_reduce_engine(
            self._ptr,
            inp,
            out,
            self.buffer_ptrs[self.rank],
            self.max_size,
            tile_numel,
            producer_blocks,
            reducer_blocks,
            compute_iters,
        )
        return out

    def tile_runtime_wait_reduce(
        self,
        staging: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        tile_numel: int = 128,
        reducer_blocks: int = 0,
    ) -> torch.Tensor:
        if self.disabled:
            raise RuntimeError("custom allreduce is disabled")
        if out is None:
            out = torch.empty_like(staging)
        ops.tile_runtime_wait_reduce(
            self._ptr,
            staging,
            out,
            tile_numel,
            reducer_blocks,
        )
        return out

    def custom_tile_runtime_all_reduce(
        self, input: torch.Tensor
    ) -> torch.Tensor | None:
        """Graph-compatible TileRT-style TP2 all-reduce experiment.

        This intentionally stays narrower than the normal custom all-reduce
        dispatcher. It is used only by SM70 AWQ MLP down_proj experiments so
        we can compare the tile-runtime substrate against the production AR
        path without changing global dispatch behavior.
        """
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self.world_size != 2:
            return None
        if input.dtype not in (torch.float16, torch.float32):
            return None

        tile_numel = envs.VLLM_SM70_AWQ_MLP_DOWN_TILE_AR_TILE_NUMEL
        if tile_numel <= 0 or input.numel() != tile_numel:
            return None

        mode = envs.VLLM_SM70_AWQ_MLP_DOWN_TILE_AR_MODE
        if mode == "inline":
            return self.tile_runtime_all_reduce(
                input,
                tile_numel=tile_numel,
                engine_blocks=envs.VLLM_SM70_AWQ_MLP_DOWN_TILE_AR_ENGINE_BLOCKS,
            )
        if mode == "engine":
            return self.tile_runtime_all_reduce_engine(
                input,
                tile_numel=tile_numel,
                producer_blocks=envs.VLLM_SM70_AWQ_MLP_DOWN_TILE_AR_PRODUCER_BLOCKS,
                reducer_blocks=envs.VLLM_SM70_AWQ_MLP_DOWN_TILE_AR_REDUCER_BLOCKS,
            )

        logger.warning_once(
            "Ignoring invalid VLLM_SM70_AWQ_MLP_DOWN_TILE_AR_MODE=%s; "
            "expected 'inline' or 'engine'.",
            mode,
            scope="global",
        )
        return None

    def awq_mlp_down_tile_gemm_reduce(
        self,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        *,
        tile_numel: int,
        reducer_blocks: int,
        kernel_reducer_blocks: int,
        overlap: bool,
    ) -> torch.Tensor | None:
        if self.disabled:
            return None
        if self.world_size != 2:
            return None
        if input.dtype != torch.float16 or input.dim() != 2 or input.size(0) != 1:
            return None
        out_features = qweight.shape[-1] * 8
        if out_features <= 0:
            return None
        if tile_numel <= 0 or out_features % tile_numel != 0:
            return None
        if out_features // tile_numel > 64:
            return None
        if input.stride(-1) != 1:
            input = input.contiguous()

        from vllm import _sm70_ops as sm70_ops

        staging = torch.empty(
            (input.size(0), out_features), dtype=input.dtype, device=input.device
        )
        if self._IS_CAPTURING and torch.cuda.is_current_stream_capturing():
            out = torch.empty_like(staging)
            sm70_ops.awq_gemm_sm70_out_tile_reduce(
                out,
                staging,
                input,
                qweight,
                scales,
                group_size,
                k_ld,
                q_ld,
                self._ptr,
                tile_numel,
                reducer_blocks,
                kernel_reducer_blocks,
                overlap,
            )
            return out

        sm70_ops.awq_gemm_sm70_out(
            staging,
            input,
            qweight,
            scales,
            group_size,
            k_ld,
            q_ld,
            False,
        )
        return self.all_reduce(staging, registered=False)

    def custom_top1_argmax(self, input_pair: torch.Tensor) -> torch.Tensor | None:
        if self.disabled:
            return None
        if input_pair.dtype != torch.float32 or input_pair.numel() != 2:
            return None
        if not is_weak_contiguous(input_pair):
            return None
        if self.world_size != 2 and not self.fully_connected:
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.top1_argmax(input_pair, registered=True)
            # Graph warmup still consumes the sampled token. Returning an
            # uninitialized placeholder here can change the dummy decode
            # sequence or even produce an invalid token id, so use the exact
            # all-gather fallback until the real CUDA graph capture starts.
            return None
        return self.top1_argmax(input_pair, registered=False)

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor | None:
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                if _AR_EVT and _AR_EVT_OK[0] and len(_AR_GRAPH_PAIRS) < 512:
                    pre, post = _ar_evt_new_pair()
                    if pre is not None:
                        try:
                            pre.record()
                            out = self.all_reduce(input, registered=True)
                            post.record()
                            _AR_GRAPH_PAIRS.append((pre, post, tuple(input.shape)))
                            n = len(_AR_GRAPH_PAIRS)
                            if n == 1 or n % 128 == 0:
                                print(f"[ar-evt] baked {n} graph pairs",
                                      flush=True)
                            if not _AR_THREAD[0]:
                                _AR_THREAD[0] = True
                                _ar_evt_thread()
                            return out
                        except Exception as exc:
                            _AR_EVT_OK[0] = False
                            print(f"[ar-evt] DISABLED at capture: "
                                  f"{type(exc).__name__}: {exc}",
                                  flush=True)
                return self.all_reduce(input, registered=True)
            # Graph warmup can still feed persistent model state (KV, SSM, or
            # CUDA graph metadata buffers). Returning an uninitialized tensor
            # here can poison the state captured immediately afterwards. Keep
            # the same out-of-place allocation pattern, but compute the real
            # reduction through the eager registered-buffer path.
            return self.all_reduce(input, registered=False)
        else:
            # Note: outside of cuda graph context, custom allreduce incurs a
            # cost of cudaMemcpy, which should be small (<=1% of overall
            # latency) compared to the performance gain of using custom kernels
            if _AR_EVT and _AR_EVT_OK[0]:
                _AR_CALLS[0] += 1
                pre, post = _ar_evt_new_pair()
                if pre is not None:
                    try:
                        pre.record()
                        out = self.all_reduce(input, registered=False)
                        post.record()
                        _AR_EAGER_PAIRS.append((pre, post))
                        if _AR_CALLS[0] % 2000 == 0:
                            _ar_evt_harvest()
                        return out
                    except Exception:
                        _AR_EVT_OK[0] = False
            return self.all_reduce(input, registered=False)

    def custom_all_reduce_sum2(
        self, input_a: torch.Tensor, input_b: torch.Tensor
    ) -> torch.Tensor | None:
        if self.disabled or not self.should_custom_ar(input_a):
            return None
        if input_a.shape != input_b.shape or input_a.dtype != input_b.dtype:
            return None
        if not is_weak_contiguous(input_b):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce_sum2(input_a, input_b)
            return self.all_reduce(input_a + input_b, registered=False)
        return None

    def close(self):
        if not self.disabled and self._ptr:
            if ops is not None:
                ops.dispose(self._ptr)
            self._ptr = 0
            self.free_shared_buffer(self.meta_ptrs, rank=self.rank)
            self.free_shared_buffer(self.buffer_ptrs, rank=self.rank)

    def __del__(self):
        self.close()

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int,
        group: ProcessGroup | None = None,
        uncached: bool | None = False,
    ) -> list[int]:
        pointer, handle = ops.allocate_shared_buffer_and_handle(size_in_bytes)

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: list[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer)  # type: ignore
            else:
                pointers.append(ops.open_mem_handle(h))
        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: list[int],
        group: ProcessGroup | None = None,
        rank: int | None = None,
    ) -> None:
        if rank is None:
            rank = dist.get_rank(group=group)
        if ops is not None:
            ops.free_shared_buffer(pointers[rank])
