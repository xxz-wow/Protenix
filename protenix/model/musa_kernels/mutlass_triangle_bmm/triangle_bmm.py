# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from functools import lru_cache
from typing import Optional

import torch

from protenix.model.extension_loader import load_prebuilt_extension
from protenix.model import tri_timing


_TRIANGLE_BMM_SCRATCH = {}
_TORCH_BMM_SCRATCH = {}


@lru_cache(maxsize=1)
def _load_extension():
    return load_prebuilt_extension(
        "protenix_mutlass_triangle_bmm_v5",
        os.path.dirname(__file__),
    )


def pack_triangle_projections(
    a_gate: torch.Tensor,
    a_projection: torch.Tensor,
    b_gate: torch.Tensor,
    b_projection: torch.Tensor,
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    outgoing: bool,
) -> None:
    """Fuse sigmoid gates with aligned triangle projection packing."""
    _load_extension().pack_triangle_projections(
        a_gate,
        a_projection,
        b_gate,
        b_projection,
        a_packed,
        b_packed,
        outgoing,
    )


def triangle_bmm_from_projections(
    a_gate: torch.Tensor,
    a_projection: torch.Tensor,
    b_gate: torch.Tensor,
    b_projection: torch.Tensor,
    outgoing: bool,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Fuse gates/layout packing and run the Triangle BMM kernel.

    Backend selection: explicit ``backend`` argument wins, then
    PROTENIX_TRIMUL_BACKEND (default "pack").
      - "pack": pack the gated projections into aligned [C, pad(N), pad(N)]
        scratch, run the MUTLASS BF16 batched GEMM, and return its channel-first
        [C, N, N] result with the GEMM's storage strides; the model-side
        LayerNorm performs the channel-first-to-channel-last conversion.
      - "torch_bmm": apply gates in Torch, flatten channels into the BMM
        batch dimension, and run ``torch.bmm``.
      - "mute": tiled MuteTileTriangleProjectionBMM; caller supplies the four
        ungated Linear outputs, the kernel fuses sigmoid(gate)*projection.
      - "direct": same C++ entry as "mute" (legacy name).
    """
    choice = backend or os.getenv("PROTENIX_TRIMUL_BACKEND", "pack")
    if choice == "torch_bmm":
        return torch_bmm_triangle_projections(
            a_gate,
            a_projection,
            b_gate,
            b_projection,
            outgoing,
        )
    if choice in ("direct", "mute"):
        return tri_timing.record(
            "trimul",
            _load_extension().triangle_projections_bmm,
            a_gate,
            a_projection,
            b_gate,
            b_projection,
            outgoing,
        )
    if choice != "pack":
        raise ValueError(
            f"unknown trimul backend {choice!r}; expected 'pack', 'torch_bmm', "
            "'direct', or 'mute'"
        )
    size = a_gate.shape[0]
    channels = a_gate.shape[2]
    padded_size = (size + 7) // 8 * 8
    stream = torch.musa.current_stream(a_gate.device)
    key = (
        a_gate.device,
        a_gate.dtype,
        channels,
        padded_size,
        stream.musa_stream,
    )
    scratch = _TRIANGLE_BMM_SCRATCH.get(key)
    if scratch is None:
        # The pack kernel writes only the [N, N] region, so the scratch must
        # be zero-initialized for any GEMM tile over-read into the pad region;
        # the GEMM never writes the packed buffers, so the zeros persist
        # across calls and the scratch can be reused.
        shape = (channels, padded_size, padded_size)
        scratch = (
            torch.zeros(shape, device=a_gate.device, dtype=a_gate.dtype),
            torch.zeros(shape, device=a_gate.device, dtype=a_gate.dtype),
        )
        _TRIANGLE_BMM_SCRATCH[key] = scratch
    a_packed, b_packed = scratch
    pack_triangle_projections(
        a_gate,
        a_projection,
        b_gate,
        b_projection,
        a_packed,
        b_packed,
        outgoing,
    )
    gemm = triangle_bmm(a_packed, b_packed, size, size, size)
    # Keep MUTLASS's channel-first [C, N, N] output and its storage strides
    # (including any alignment padding).  The model boundary uses the
    # channel-first LayerNorm kernel to normalize and write the final
    # contiguous channel-last [N, N, C] tensor.
    return gemm


def triangle_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    m: int,
    n: int,
    k: int,
) -> torch.Tensor:
    """Run padded MP31 MUTLASS BF16 strided-batched GEMM."""
    padded_m = (m + 7) // 8 * 8
    padded_n = (n + 7) // 8 * 8
    padded_k = (k + 7) // 8 * 8
    if (
        padded_m > a.size(1)
        or padded_n > b.size(2)
        or padded_k > a.size(2)
        or padded_k > b.size(1)
    ):
        raise ValueError(
            "triangle_bmm padded problem shape exceeds packed buffer shape"
        )
    output = tri_timing.record(
        "trimul", _load_extension().triangle_bmm, a, b, padded_m, padded_n, padded_k
    )
    if padded_m != m or padded_n != n:
        output = output.narrow(1, 0, m).narrow(2, 0, n)
    return output


def _torch_bmm_scratch(a_gate: torch.Tensor, size: int, channels: int):
    """Return the reusable worker stream and first-partial buffer."""
    current_stream = torch.musa.current_stream(a_gate.device)
    key = (
        a_gate.device,
        a_gate.dtype,
        channels,
        size,
        current_stream.musa_stream,
    )
    state = _TORCH_BMM_SCRATCH.get(key)
    if state is None:
        state = {
            "stream": torch.musa.Stream(device=a_gate.device),
            "ready": torch.musa.Event(),
            "partial": torch.empty(
                (channels, size, size),
                device=a_gate.device,
                dtype=a_gate.dtype,
            ),
        }
        _TORCH_BMM_SCRATCH[key] = state
    return current_stream, state


def _torch_bmm_prepare_chunk(
    gate: torch.Tensor,
    projection: torch.Tensor,
    start: int,
    length: int,
    transpose_output: bool,
    incoming: bool,
) -> torch.Tensor:
    """Gate one K chunk and pack it as a channel-batched matrix."""
    if incoming:
        chunk = slice(start, start + length)
        gated = torch.sigmoid(gate[chunk, :, :]) * projection[chunk, :, :]
        packed = gated.permute(2, 0, 1).contiguous()
    else:
        chunk = slice(start, start + length)
        gated = torch.sigmoid(gate[:, chunk, :]) * projection[:, chunk, :]
        packed = gated.permute(2, 0, 1).contiguous()
    if transpose_output:
        packed = packed.transpose(1, 2).contiguous()
    return packed


def _torch_bmm_triangle_projections_streamed(
    a_gate: torch.Tensor,
    a_projection: torch.Tensor,
    b_gate: torch.Tensor,
    b_projection: torch.Tensor,
    outgoing: bool,
) -> torch.Tensor:
    """Overlap second-chunk gating with the first chunk's MUSA BMM."""
    size = a_gate.shape[0]
    channels = a_gate.shape[2]
    split = size // 2

    current_stream, state = _torch_bmm_scratch(a_gate, size, channels)
    a_first = _torch_bmm_prepare_chunk(
        a_gate, a_projection, 0, split, not outgoing, not outgoing
    )
    b_first = _torch_bmm_prepare_chunk(
        b_gate, b_projection, 0, split, outgoing, not outgoing
    )
    state["ready"].record(current_stream)
    a_first.record_stream(state["stream"])
    b_first.record_stream(state["stream"])
    with torch.musa.stream(state["stream"]):
        state["stream"].wait_event(state["ready"])
        tri_timing.record(
            "trimul",
            torch.bmm,
            a_first,
            b_first,
            out=state["partial"],
        )

    a_second = _torch_bmm_prepare_chunk(
        a_gate, a_projection, split, size - split, not outgoing, not outgoing
    )
    b_second = _torch_bmm_prepare_chunk(
        b_gate, b_projection, split, size - split, outgoing, not outgoing
    )
    second = torch.bmm(a_second, b_second)
    current_stream.wait_stream(state["stream"])
    second.add_(state["partial"])
    return second


def torch_bmm_triangle_projections(
    a_gate: torch.Tensor,
    a_projection: torch.Tensor,
    b_gate: torch.Tensor,
    b_projection: torch.Tensor,
    outgoing: bool,
) -> torch.Tensor:
    """Run gated triangle multiplication as two K-split batched matmuls.

    On MUSA, the first partial result uses a reusable buffer and a persistent
    worker stream while the current stream prepares the second K chunk.  The
    CPU path keeps the same decomposition without MUSA stream management so
    the helper remains directly testable.
    """
    if a_gate.ndim != 3 or a_projection.ndim != 3:
        raise ValueError("torch_bmm triangle backend expects [N, N, C] projections")
    if (
        a_gate.shape != a_projection.shape
        or b_gate.shape != b_projection.shape
        or a_gate.shape != b_gate.shape
    ):
        raise ValueError("triangle gate and projection tensors must have matching shapes")
    if a_gate.shape[0] != a_gate.shape[1]:
        raise ValueError("torch_bmm triangle backend expects square pair matrices")

    size = a_gate.shape[0]
    channels = a_gate.shape[2]
    split = size // 2
    if a_gate.device.type != "musa":
        a_first = _torch_bmm_prepare_chunk(
            a_gate, a_projection, 0, split, not outgoing, not outgoing
        )
        b_first = _torch_bmm_prepare_chunk(
            b_gate, b_projection, 0, split, outgoing, not outgoing
        )
        a_second = _torch_bmm_prepare_chunk(
            a_gate, a_projection, split, size - split, not outgoing, not outgoing
        )
        b_second = _torch_bmm_prepare_chunk(
            b_gate, b_projection, split, size - split, outgoing, not outgoing
        )
        result = torch.bmm(a_first, b_first) + torch.bmm(a_second, b_second)
        return result.reshape(channels, size, size).permute(1, 2, 0).contiguous()

    result = _torch_bmm_triangle_projections_streamed(
        a_gate, a_projection, b_gate, b_projection, outgoing
    )
    return result.permute(1, 2, 0).contiguous()
