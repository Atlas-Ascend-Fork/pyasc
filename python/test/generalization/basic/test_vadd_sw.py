# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest

import asc
import asc.runtime.config as config
import asc.lib.runtime as rt

try:
    import torch
except ModuleNotFoundError:
    pytest.skip("torch is not installed", allow_module_level=True)


@asc.jit
def vadd_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress, block_length: asc.ConstExpr[int],
                buffer_num: asc.ConstExpr[int], tile_length: asc.ConstExpr[int], tile_num: asc.ConstExpr[int]):
    offset = asc.get_block_idx() * block_length
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    z_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + offset)
    y_gm.set_global_buffer(y + offset)
    z_gm.set_global_buffer(z + offset)

    pipe = asc.TPipe()
    in_queue_x = asc.TQue(asc.TPosition.VECIN, buffer_num)
    in_queue_y = asc.TQue(asc.TPosition.VECIN, buffer_num)
    out_queue_z = asc.TQue(asc.TPosition.VECOUT, buffer_num)
    tmp_queue = asc.TQue(asc.TPosition.VECCALC, buffer_num)

    for i in range(tile_num):
        pipe.destroy()
        pipe.init()
        pipe.init_buffer(que=in_queue_x, num=buffer_num, len=tile_length * x.dtype.sizeof())
        pipe.init_buffer(que=in_queue_y, num=buffer_num, len=tile_length * y.dtype.sizeof())
        pipe.init_buffer(que=out_queue_z, num=buffer_num, len=tile_length * z.dtype.sizeof())
        pipe.init_buffer(que=tmp_queue, num=buffer_num, len=tile_length * z.dtype.sizeof())
        global_pipe = asc.get_tpipe_ptr()
        compute(global_pipe, i, z_gm, x_gm, y_gm, in_queue_x, in_queue_y, out_queue_z, tmp_queue, tile_length)
        pipe.reset()


@asc.jit
def compute(global_pipe: asc.TPipe, i: int, z_gm: asc.GlobalTensor, x_gm: asc.GlobalAddress, y_gm: asc.GlobalAddress,
            in_queue_x: asc.TQue, in_queue_y: asc.TQue, out_queue_z: asc.TQue, tmp_queue: asc.TQue,
            tile_length: asc.ConstExpr[int]):
    x_local = in_queue_x.alloc_tensor(x_gm.dtype)
    y_local = in_queue_y.alloc_tensor(y_gm.dtype)
    z_local = out_queue_z.alloc_tensor(z_gm.dtype)

    _ = z_gm.get_phy_addr()
    _ = x_local.get_phy_addr()
    x_local.set_buffer_len(tile_length * 4)
    x_local.set_size(tile_length)

    asc.data_copy(x_local, x_gm[i * tile_length:], count=tile_length)
    asc.data_copy(y_local, y_gm[i * tile_length:], count=tile_length)

    eventid1 = global_pipe.alloc_event_id(event=asc.HardEvent.MTE2_V)
    asc.set_flag(event=asc.HardEvent.MTE2_V, event_id=eventid1)
    asc.wait_flag(event=asc.HardEvent.MTE2_V, event_id=eventid1)
    global_pipe.release_event_id(id=eventid1, event=asc.HardEvent.MTE2_V)

    src_pos = x_local.get_position()
    _ = x_local.get_length()
    _ = x_local.get_size()

    x_tag = 13  # set custom tag
    x_local.set_user_tag(x_tag)
    z_tag = x_local.get_user_tag()

    tmp_dtype = asc.DataType("int32")
    tmp_local = tmp_queue.alloc_tensor(tmp_dtype)
    new_tmp = tmp_local.reinterpret_cast(x_gm.dtype)

    if z_tag == 13 and src_pos == asc.TPosition.VECIN:
        asc.add(z_local, x_local, y_local, count=tile_length)

    eventid2 = global_pipe.fetch_event_id(event=asc.HardEvent.V_MTE3)
    asc.set_flag(event=asc.HardEvent.V_MTE3, event_id=eventid2)
    asc.wait_flag(event=asc.HardEvent.V_MTE3, event_id=eventid2)

    asc.data_copy(z_gm[i * tile_length:], z_local, count=tile_length)

    new_tmp.set_addr_with_offset(tmp_local, 16)
    _ = new_tmp.get_size()

    in_queue_x.free_tensor(x_local)
    in_queue_y.free_tensor(y_local)
    out_queue_z.free_tensor(z_local)
    tmp_queue.free_tensor(tmp_local)


def vadd_launch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    z = torch.zeros_like(x)
    total_length = z.numel()
    use_core_num = 16
    block_length = (total_length + use_core_num - 1) // use_core_num
    tile_length = 512
    tile_num = (block_length + tile_length - 1) // tile_length
    buffer_num = 1
    vadd_kernel[use_core_num, rt.current_stream()](x, y, z, block_length, buffer_num, tile_length, tile_num)
    return z


param_list = [
    [torch.float32, (1000, )],
    [torch.float32, (1, )],
    [torch.float32, (9999, )],
    [torch.float16, (2048, )],
    [torch.int32, (8192, )],
    [torch.int16, (8192, )],
    [torch.int32, (153, 834)],
]

BACKENDS = [
    # config.Backend.Model,
    config.Backend.NPU,
]


@pytest.mark.parametrize("dtype, size", param_list)
@pytest.mark.parametrize("backend", BACKENDS)
def test_vadd_sw(dtype, size, backend: config.Backend):
    config.set_platform(backend)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    if dtype in {torch.float16, torch.float32}:
        x = torch.randn(size, dtype=dtype, device=device)
        y = torch.randn(size, dtype=dtype, device=device)
    else:
        x = torch.randint(-100, 99, size, dtype=dtype, device=device)
        y = torch.randint(-100, 99, size, dtype=dtype, device=device)
    z = vadd_launch(x, y)
    assert torch.allclose(z, x + y)
