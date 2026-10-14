# Copyright (c) 2026 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import argparse
import torch

import asc
import asc.runtime.config as config
import asc.lib.runtime as rt
from asc.lib.runtime.support import DeviceModuleType, DeviceInfoType

logging.basicConfig(level=logging.INFO)

BUFFER_NUM = 2
DATABLOCK_BYTES = 32
PREFERRED_COPY_BYTES = 2048  # Larger tile for better bandwidth utilization
FALLBACK_COPY_BYTES = 1024  # Reduced tile when data volume is insufficient


def _gen_core_candidates(max_core_num: int):
    candidates = []
    c = 1
    while c <= max_core_num:
        candidates.append(c)
        c *= 2
    if candidates[-1] != max_core_num:
        candidates.append(max_core_num)
    return tuple(candidates)


def get_max_core_num() -> int:
    return max(1, int(rt.device_info(DeviceModuleType.RT_MODULE_TYPE_VECTOR_CORE, DeviceInfoType.INFO_TYPE_CORE_NUM)))


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def compute_launch_params(total_length: int, dtype_size: int, max_core_num: int = None):
    if max_core_num is None:
        max_core_num = get_max_core_num()
    vec_align_elems = max(1, DATABLOCK_BYTES // dtype_size)
    preferred_tile = max(vec_align_elems, PREFERRED_COPY_BYTES // dtype_size)
    fallback_tile = max(vec_align_elems, FALLBACK_COPY_BYTES // dtype_size)

    tile_length = preferred_tile if total_length >= preferred_tile else fallback_tile
    needed_cores = min(max_core_num, max(1, ceil_div(total_length, tile_length)))
    candidates = _gen_core_candidates(max_core_num)
    effective_cores = candidates[-1]
    for candidate in candidates:
        if needed_cores <= candidate:
            effective_cores = candidate
            break
    block_length_raw = ceil_div(total_length, effective_cores)
    total_tiles = max(1, ceil_div(block_length_raw, tile_length))
    block_length = total_tiles * tile_length
    return effective_cores, block_length, tile_length


def pad_flat_tensor(x: torch.Tensor, padded_len: int) -> torch.Tensor:
    flat = x.reshape(-1)
    if padded_len <= flat.numel():
        return flat
    padded = flat.new_empty(padded_len)
    padded[:flat.numel()] = flat
    return padded


def empty_padded_like(x: torch.Tensor, padded_len: int) -> torch.Tensor:
    flat = x.reshape(-1)
    if padded_len <= flat.numel():
        return torch.empty_like(flat)
    return torch.empty(padded_len, dtype=x.dtype, device=x.device)


@asc.jit(kernel_type=config.KernelType.AIV_ONLY)
def swiglu_kernel(fused: asc.GlobalAddress, y: asc.GlobalAddress, block_length: asc.ConstExpr[int],
                  tile_length: asc.ConstExpr[int], half_len: asc.ConstExpr[int]):

    offset = asc.get_block_idx() * block_length
    fused_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    fused_gm.set_global_buffer(fused + offset)
    y_gm.set_global_buffer(y + offset)

    pipe = asc.TPipe()
    in_queue_gate = asc.TQue(asc.TPosition.VECIN, 1)
    in_queue_up = asc.TQue(asc.TPosition.VECIN, 1)
    out_queue = asc.TQue(asc.TPosition.VECOUT, 1)

    pipe.init_buffer(que=in_queue_gate, num=BUFFER_NUM, len=tile_length * fused.dtype.sizeof())
    pipe.init_buffer(que=in_queue_up, num=BUFFER_NUM, len=tile_length * fused.dtype.sizeof())
    pipe.init_buffer(que=out_queue, num=BUFFER_NUM, len=tile_length * fused.dtype.sizeof())

    total_tiles = block_length // tile_length
    for i in asc.range(total_tiles):
        gate_local = in_queue_gate.alloc_tensor(fused_gm.dtype)
        up_local = in_queue_up.alloc_tensor(fused_gm.dtype)
        asc.data_copy(gate_local, fused_gm[i * tile_length:], count=tile_length)
        asc.data_copy(up_local, fused_gm[half_len + i * tile_length:], count=tile_length)
        in_queue_gate.enque(gate_local)
        in_queue_up.enque(up_local)

        gate_local = in_queue_gate.deque(fused_gm.dtype)
        up_local = in_queue_up.deque(fused_gm.dtype)
        y_local = out_queue.alloc_tensor(fused_gm.dtype)

        asc.adv.swiglu(y_local, up_local, gate_local, cal_count=tile_length)

        in_queue_gate.free_tensor(gate_local)
        in_queue_up.free_tensor(up_local)
        out_queue.enque(y_local)

        y_local = out_queue.deque(fused_gm.dtype)
        asc.data_copy(y_gm[i * tile_length:], y_local, count=tile_length)
        out_queue.free_tensor(y_local)


def swiglu_launch(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    total_length = gate.numel()
    effective_cores, block_length, tile_length = compute_launch_params(total_length, gate.element_size())
    padded_len = effective_cores * block_length
    half_len = padded_len
    gate_pad = pad_flat_tensor(gate, padded_len)
    up_pad = pad_flat_tensor(up, padded_len)
    fused = torch.cat([gate_pad, up_pad], dim=0)
    y_pad = empty_padded_like(gate, padded_len)
    swiglu_kernel[effective_cores, rt.current_stream()](fused, y_pad, block_length, tile_length, half_len)
    if gate.device.type == "npu":
        torch.npu.synchronize()
    return y_pad[:total_length].reshape_as(gate)


def swiglu_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return gate * torch.sigmoid(gate) * up


def swiglu_custom(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    gate = torch.rand(2, 8, 512, dtype=torch.float32, device=device) * 4.0 - 2.0
    up = torch.rand(2, 8, 512, dtype=torch.float32, device=device) * 4.0 - 2.0
    y = swiglu_launch(gate, up)
    expected = swiglu_reference(gate, up)
    assert torch.allclose(y, expected, atol=1e-3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=str, default="Model", help="backend to run")
    parser.add_argument("-v", type=str, default=None, help="platform to run")
    args = parser.parse_args()
    backend = args.r
    platform = args.v
    if backend not in config.Backend.__members__:
        raise ValueError(f"Unsupported Backend! Supported: {list(config.Backend.__members__.keys())}")
    backend = config.Backend(backend)
    if platform is not None:
        platform_values = [platform.value for platform in config.Platform]
        if platform not in platform_values:
            raise ValueError(f"Unsupported Platform! Supported: {platform_values}")
        platform = config.Platform(platform)
    logging.info("[INFO] start process sample swiglu.")
    swiglu_custom(backend, platform)
    logging.info("[INFO] Sample swiglu run success.")
