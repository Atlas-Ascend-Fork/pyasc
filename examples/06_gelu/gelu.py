# Copyright (c) 2026 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
import logging
import argparse
import torch

import asc
import asc.runtime.config as config
import asc.lib.runtime as rt

logging.basicConfig(level=logging.INFO)

USE_CORE_NUM = 8
BUFFER_NUM = 2
DATABLOCK_BYTES = 32
PREFERRED_COPY_BYTES = 512  # Optimal DMA transfer size for bandwidth utilization
FALLBACK_COPY_BYTES = 256  # Reduced DMA transfer size when data volume is insufficient
CORE_CANDIDATES = (1, 2, 4, USE_CORE_NUM)

GELU_CUBIC_COEFF = 0.044715
GELU_TANH_SCALE = math.sqrt(2.0 / math.pi)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def compute_launch_params(total_length: int, dtype_size: int):
    vec_align_elems = max(1, DATABLOCK_BYTES // dtype_size)
    preferred_tile = max(vec_align_elems, PREFERRED_COPY_BYTES // dtype_size)
    fallback_tile = max(vec_align_elems, FALLBACK_COPY_BYTES // dtype_size)

    # Use a DMA-friendly copy block first, then derive tiles per core from it.
    tile_length = preferred_tile if total_length >= preferred_tile else fallback_tile
    needed_cores = min(USE_CORE_NUM, max(1, ceil_div(total_length, tile_length)))
    effective_cores = USE_CORE_NUM
    for candidate in CORE_CANDIDATES:
        if needed_cores <= candidate:
            effective_cores = candidate
            break
    block_length_raw = ceil_div(total_length, effective_cores)
    total_tiles = max(1, ceil_div(block_length_raw, tile_length))
    block_length = total_tiles * tile_length
    return effective_cores, block_length, tile_length


@asc.jit(kernel_type=config.KernelType.AIV_ONLY)
def gelu_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, block_length: asc.ConstExpr[int],
                tile_length: asc.ConstExpr[int]):

    offset = asc.get_block_idx() * block_length
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + offset)
    y_gm.set_global_buffer(y + offset)

    pipe = asc.TPipe()
    in_queue = asc.TQue(asc.TPosition.VECIN, 1)
    out_queue = asc.TQue(asc.TPosition.VECOUT, 1)
    tmp_buf = asc.TBuf(asc.TPosition.VECCALC)

    pipe.init_buffer(que=in_queue, num=BUFFER_NUM, len=tile_length * x.dtype.sizeof())
    pipe.init_buffer(que=out_queue, num=BUFFER_NUM, len=tile_length * y.dtype.sizeof())
    pipe.init_buffer(buf=tmp_buf, len=tile_length * x.dtype.sizeof())

    total_tiles = block_length // tile_length
    for i in asc.range(total_tiles):
        gelu_copy_in(i, x_gm, in_queue, tile_length)
        gelu_compute(y_gm, in_queue, out_queue, tmp_buf, tile_length)
        gelu_copy_out(i, y_gm, out_queue, tile_length)


@asc.jit
def gelu_copy_in(i: int, x_gm: asc.GlobalAddress, in_queue: asc.TQue, tile_length: asc.ConstExpr[int]):
    x_local = in_queue.alloc_tensor(x_gm.dtype)
    asc.data_copy(x_local, x_gm[i * tile_length:], count=tile_length)
    in_queue.enque(x_local)


@asc.jit
def gelu_compute(y_gm: asc.GlobalTensor, in_queue: asc.TQue, out_queue: asc.TQue, tmp_buf: asc.TBuf,
                 tile_length: asc.ConstExpr[int]):
    x_local = in_queue.deque(y_gm.dtype)
    y_local = out_queue.alloc_tensor(y_gm.dtype)
    tmp = tmp_buf.get(y_gm.dtype)

    asc.mul(tmp, x_local, x_local, count=tile_length)
    asc.pipe_barrier(asc.PipeID.PIPE_V)
    asc.mul(tmp, tmp, x_local, count=tile_length)
    asc.pipe_barrier(asc.PipeID.PIPE_V)
    asc.muls(tmp, tmp, GELU_CUBIC_COEFF, count=tile_length)
    asc.pipe_barrier(asc.PipeID.PIPE_V)
    asc.add(tmp, tmp, x_local, count=tile_length)
    asc.pipe_barrier(asc.PipeID.PIPE_V)
    asc.muls(tmp, tmp, GELU_TANH_SCALE, count=tile_length)
    asc.pipe_barrier(asc.PipeID.PIPE_V)
    asc.adv.tanh(tmp, tmp, count=tile_length)
    asc.pipe_barrier(asc.PipeID.PIPE_V)
    asc.adds(tmp, tmp, 1.0, count=tile_length)
    asc.pipe_barrier(asc.PipeID.PIPE_V)
    asc.muls(tmp, tmp, 0.5, count=tile_length)
    asc.pipe_barrier(asc.PipeID.PIPE_V)
    asc.mul(y_local, x_local, tmp, count=tile_length)

    out_queue.enque(y_local)
    in_queue.free_tensor(x_local)


@asc.jit
def gelu_copy_out(i: int, y_gm: asc.GlobalTensor, out_queue: asc.TQue, tile_length: asc.ConstExpr[int]):
    y_local = out_queue.deque(y_gm.dtype)
    asc.data_copy(y_gm[i * tile_length:], y_local, count=tile_length)
    out_queue.free_tensor(y_local)


def gelu_launch(x: torch.Tensor) -> torch.Tensor:
    total_length = x.numel()
    effective_cores, block_length, tile_length = compute_launch_params(total_length, x.element_size())

    # Pad input and launch
    padded_len = effective_cores * block_length
    x_flat = x.reshape(-1)
    if padded_len > total_length:
        x_pad = torch.zeros(padded_len, dtype=x.dtype, device=x.device)
        x_pad[:total_length] = x_flat
        y_pad = torch.zeros(padded_len, dtype=x.dtype, device=x.device)
    else:
        x_pad = x_flat
        y_pad = torch.zeros(padded_len, dtype=x.dtype, device=x.device)
    gelu_kernel[effective_cores, rt.current_stream()](x_pad, y_pad, block_length, tile_length)
    if x.device.type == "npu":
        torch.npu.synchronize()
    y = y_pad[:total_length].reshape_as(x)
    return y


def gelu_reference(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(GELU_TANH_SCALE * (x + GELU_CUBIC_COEFF * x.pow(3))))


def gelu_custom(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    size = 8 * 2048
    x = torch.rand(size, dtype=torch.float32, device=device) * 4.0 - 2.0
    y = gelu_launch(x)
    expected = gelu_reference(x)
    assert torch.allclose(y, expected, atol=1e-3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=str, default="Model", help="backend to run")
    parser.add_argument("-v", type=str, default=None, help="platform to run")
    args = parser.parse_args()
    backend = args.r
    platform = args.v
    if backend not in config.Backend.__members__:
        raise ValueError("Unsupported Backend! Supported: ['Model', 'NPU']")
    backend = config.Backend(backend)
    if platform is not None:
        platform_values = [platform.value for platform in config.Platform]
        if platform not in platform_values:
            raise ValueError(f"Unsupported Platform! Supported: {platform_values}")
        platform = config.Platform(platform)
    logging.info("[INFO] start process sample gelu.")
    gelu_custom(backend, platform)
    logging.info("[INFO] Sample gelu run success.")
