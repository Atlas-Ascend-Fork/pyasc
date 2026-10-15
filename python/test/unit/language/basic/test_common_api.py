# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc
from asc.runtime import config
from asc.runtime.jit import MockTensor
import pytest


def setup_function():
    config.set_platform(config.Backend.Model, check=False)


def test_aipp_functions_single_src(mock_launcher_run):

    @asc.jit
    def kernel_aipp_single_src(x: asc.GlobalAddress) -> None:
        rgb_gm = asc.GlobalTensor()
        rgb_gm.set_global_buffer(x)

        swap_settings = asc.AippSwapParams(is_swap_rb=True)
        aipp_config = asc.AippParams(dtype=asc.int8, swap_params=swap_settings)

        asc.set_aipp_functions(rgb_gm, asc.AippInputFormat.RGB888_U8, aipp_config)

    x = MockTensor(asc.int8)
    kernel_aipp_single_src[1](x)
    assert mock_launcher_run.call_count == 1


def test_aipp_functions_dual_src(mock_launcher_run):

    @asc.jit
    def kernel_aipp_dual_src(y: asc.GlobalAddress, uv: asc.GlobalAddress) -> None:
        y_gm = asc.GlobalTensor()
        uv_gm = asc.GlobalTensor()
        y_gm.set_global_buffer(y)
        uv_gm.set_global_buffer(uv)

        dtc_settings = asc.AippDataTypeConvParams(dtc_mean_ch0=128)
        aipp_config = asc.AippParams(dtype=asc.float16, dtc_params=dtc_settings)

        asc.set_aipp_functions(y_gm, uv_gm, asc.AippInputFormat.YUV420SP_U8, aipp_config)

    y = MockTensor(asc.float16)
    uv = MockTensor(asc.float16)
    kernel_aipp_dual_src[1](y, uv)
    assert mock_launcher_run.call_count == 1


def test_copy(mock_launcher_run):

    @asc.jit
    def kernel_copy() -> None:
        dst_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        src_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        params = asc.CopyRepeatParams(1, 1, 8, 8)

        asc.copy(dst=dst_local, src=src_local, mask=64, repeat_time=4, repeat_params=params, is_set_mask=True)

        uint64_max = 2**64 - 1
        mask_bits = [uint64_max, uint64_max]
        asc.copy(dst=dst_local, src=src_local, mask=mask_bits, repeat_time=1, repeat_params=params, is_set_mask=True)

    kernel_copy[1]()
    assert mock_launcher_run.call_count == 1


def test_duplicate(mock_launcher_run):

    @asc.jit
    def kernel_duplicate() -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        asc.duplicate(x_local, 5, count=512)
        asc.duplicate(x_local, 5, 512, 1, 1, 1)
        asc.duplicate(x_local, 5, [2**64 - 1, 2**64 - 1], 1, 1, 1)

    kernel_duplicate[1]()
    assert mock_launcher_run.call_count == 1


def test_gatherb(mock_launcher_run):

    @asc.jit
    def kernel_gatherb():
        x_local = asc.LocalTensor(dtype=asc.uint16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        src_offset = asc.LocalTensor(dtype=asc.uint32, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.uint16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        params = asc.GatherRepeatParams(dst_blk_stride=1, dst_rep_stride=8)
        asc.gatherb(z_local, x_local, src_offset, repeat_times=1, repeat_params=params)

    kernel_gatherb[1]()
    assert mock_launcher_run.call_count == 1


def test_gather(mock_launcher_run):

    @asc.jit
    def kernel_gather() -> None:
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        src_offset = asc.LocalTensor(dtype=asc.uint32, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        asc.gather(z_local, x_local, src_offset, src_base=0, count=512)
        asc.gather(z_local, x_local, src_offset, src_base=0, mask=512, repeat_times=1, dst_rep_stride=8)
        uint64_max = 2**64 - 1
        mask_bits = [uint64_max, uint64_max]
        asc.gather(z_local, x_local, src_offset, src_base=0, mask=mask_bits, repeat_times=1, dst_rep_stride=8)

    kernel_gather[1]()
    assert mock_launcher_run.call_count == 1


def test_scatter(mock_launcher_run):

    @asc.jit
    def kernel_scatter() -> None:
        dst = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=128)
        dst_offset = asc.LocalTensor(dtype=asc.uint32, pos=asc.TPosition.VECIN, addr=0, tile_size=128)
        src = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=128)
        asc.scatter(dst, src, dst_offset, dst_base=0, count=128)
        asc.scatter(dst, src, dst_offset, dst_base=0, mask=128, repeat_times=1, src_rep_stride=8)
        uint64_max = 2**64 - 1
        mask_bits = [uint64_max, uint64_max]
        asc.scatter(dst, src, dst_offset, dst_base=0, mask=mask_bits, repeat_times=1, src_rep_stride=8)

    kernel_scatter[1]()
    assert mock_launcher_run.call_count == 1


def test_data_copy(mock_launcher_run):

    @asc.jit
    def kernel_data_copy(x: asc.GlobalAddress) -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        x_gm = asc.GlobalTensor()
        x_gm.set_global_buffer(x)
        asc.data_copy(x_gm, x_local, count=512)
        asc.data_copy(y_local, x_local, count=512)
        asc.data_copy(x_local, x_gm, count=512)
        intri_params = asc.DataCopyParams()
        asc.data_copy(x_gm, x_local, repeat_params=intri_params)
        asc.data_copy(y_local, x_local, repeat_params=intri_params)
        asc.data_copy(x_local, x_gm, repeat_params=intri_params)
        intri_params = asc.Nd2NzParams(1, 32, 32, 0, 32, 32, 1, 0)
        asc.data_copy(y_local, x_local, intri_params=intri_params)
        asc.data_copy(x_local, x_gm, intri_params=intri_params)
        intri_params = asc.Nz2NdParamsFull(1, 32, 32, 1, 32, 32, 1)
        asc.data_copy(x_gm, x_local, intri_params=intri_params)
        intri_params = asc.DataCopyCO12DstParams(16, 1, 16, 1)
        asc.data_copy(x_gm, x_local, intri_params=intri_params)
        asc.data_copy(y_local, x_local, intri_params=intri_params)

    x = MockTensor(asc.float16)
    kernel_data_copy[1](x)
    assert mock_launcher_run.call_count == 1


def test_data_sync_barrier_all(mock_launcher_run):

    @asc.jit
    def kernel_data_sync_barrier_all() -> None:
        asc.data_sync_barrier(asc.MemDsbT.ALL)

    kernel_data_sync_barrier_all[1]()
    assert mock_launcher_run.call_count == 1


def test_get_arch_version(mock_launcher_run):

    @asc.jit
    def kernel_get_arch_version(core_version: int) -> None:
        asc.get_arch_version(core_version=core_version)

    kernel_get_arch_version[1](0)
    assert mock_launcher_run.call_count == 1


def test_brcb_kernel(mock_launcher_run):

    @asc.jit
    def brcb_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        params = asc.BrcbRepeatParams(1, 8)
        asc.brcb(z_local, x_local, repeat_times=1, repeat_params=params)

    brcb_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_block_reduce_max_kernel(mock_launcher_run):

    @asc.jit
    def block_reduce_max_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.block_reduce_max(z_local, x_local, repeat=1, mask=512, dst_rep_stride=8, src_blk_stride=1, src_rep_stride=8)
        asc.block_reduce_max(z_local, x_local, repeat=1, mask=0, dst_rep_stride=8, src_blk_stride=1, src_rep_stride=8)
        int32_max = 2**31 - 1
        asc.block_reduce_max(z_local, x_local, repeat=1, mask=int32_max, dst_rep_stride=8, src_blk_stride=1,
                             src_rep_stride=8)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.block_reduce_max(z_local, x_local, repeat=1, mask=mask, dst_rep_stride=8, src_blk_stride=1,
                             src_rep_stride=8)

    block_reduce_max_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_block_reduce_min_kernel(mock_launcher_run):

    @asc.jit
    def block_reduce_min_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.block_reduce_min(z_local, x_local, repeat=1, mask=512, dst_rep_stride=8, src_blk_stride=1, src_rep_stride=8)
        asc.block_reduce_min(z_local, x_local, repeat=1, mask=0, dst_rep_stride=8, src_blk_stride=1, src_rep_stride=8)
        int32_max = 2**31 - 1
        asc.block_reduce_min(z_local, x_local, repeat=1, mask=int32_max, dst_rep_stride=8, src_blk_stride=1,
                             src_rep_stride=8)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.block_reduce_min(z_local, x_local, repeat=1, mask=mask, dst_rep_stride=8, src_blk_stride=1,
                             src_rep_stride=8)

    block_reduce_min_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_block_reduce_sum_kernel(mock_launcher_run):

    @asc.jit
    def block_reduce_sum_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.block_reduce_sum(z_local, x_local, repeat=1, mask=512, dst_rep_stride=8, src_blk_stride=1, src_rep_stride=8)
        asc.block_reduce_sum(z_local, x_local, repeat=1, mask=0, dst_rep_stride=8, src_blk_stride=1, src_rep_stride=8)
        int32_max = 2**31 - 1
        asc.block_reduce_sum(z_local, x_local, repeat=1, mask=int32_max, dst_rep_stride=8, src_blk_stride=1,
                             src_rep_stride=8)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.block_reduce_sum(z_local, x_local, repeat=1, mask=mask, dst_rep_stride=8, src_blk_stride=1,
                             src_rep_stride=8)

    block_reduce_sum_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_data_copy_pad(mock_launcher_run):

    @asc.jit
    def kernel_data_copy_pad(x: asc.GlobalAddress) -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        x_gm = asc.GlobalTensor()
        x_gm.set_global_buffer(x)
        params = asc.DataCopyParams(1, 64, 0, 0)
        pad_params = asc.DataCopyPadParams(True, 2, 4, 0)
        nd2nz_params = asc.Nd2NzParams(1, 2, 3, 4, 5, 6, 7, 8)
        ext_params = asc.DataCopyExtParams(1, 64, 0, 0, 8)
        pad_ext_params = asc.DataCopyPadExtParams(dtype=asc.float16, is_pad=True, left_padding=2, right_padding=4,
                                                  padding_value=0)
        asc.data_copy_pad(x_local, x_gm, params, pad_params)
        asc.data_copy_pad(x_gm, x_local, params)
        asc.data_copy_pad(x_local, x_local, params, nd2nz_params)
        asc.data_copy_pad(x_local, x_gm, ext_params, pad_ext_params)
        asc.data_copy_pad(x_gm, x_local, ext_params)
        asc.data_copy_pad(x_local, x_local, ext_params, nd2nz_params)

    x = MockTensor(asc.float16)
    kernel_data_copy_pad[1](x)
    assert mock_launcher_run.call_count == 1


def test_load_image_to_local(mock_launcher_run):

    @asc.jit
    def kernel_load_image_to_local() -> None:
        dst = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.A1, addr=0, tile_size=128)
        load_data_params = asc.LoadImageToLocalParams(2, 2, 0, 0, 2, 0, 0, 0, 0)
        asc.load_image_to_local(dst, load_data_params)

    kernel_load_image_to_local[1]()
    assert mock_launcher_run.call_count == 1


def test_get_block_idx(mock_launcher_run):

    @asc.jit
    def kernel_get_block_idx() -> None:
        idx = asc.get_block_idx()

    kernel_get_block_idx[1]()
    assert mock_launcher_run.call_count == 1


def test_get_block_num(mock_launcher_run):

    @asc.jit
    def kernel_get_block_num() -> None:
        num = asc.get_block_num()

    kernel_get_block_num[1]()
    assert mock_launcher_run.call_count == 1


def test_get_data_block_size_in_bytes(mock_launcher_run):

    @asc.jit
    def kernel_get_data_block_size_in_bytes() -> None:
        size = asc.get_data_block_size_in_bytes()

    kernel_get_data_block_size_in_bytes[1]()
    assert mock_launcher_run.call_count == 1


def test_get_sub_block_idx(mock_launcher_run):

    @asc.jit
    def kernel_get_sub_block_idx() -> None:
        idx = asc.get_sub_block_idx()

    kernel_get_sub_block_idx[1]()
    assert mock_launcher_run.call_count == 1


def test_get_sub_block_num(mock_launcher_run):

    @asc.jit
    def kernel_get_sub_block_num() -> None:
        num = asc.get_sub_block_num()

    kernel_get_sub_block_num[1]()
    assert mock_launcher_run.call_count == 1


def test_get_sys_workspace(mock_launcher_run):

    @asc.jit
    def kernel_get_sys_workspace() -> None:
        x = asc.get_sys_workspace()

    kernel_get_sys_workspace[1]()
    assert mock_launcher_run.call_count == 1


def test_get_task_ratio(mock_launcher_run):

    @asc.jit
    def kernel_get_task_ratio() -> None:
        ratio = asc.get_task_ratio()

    kernel_get_task_ratio[1]()
    assert mock_launcher_run.call_count == 1


def test_ascend_is_aic(mock_launcher_run):

    @asc.jit
    def kernel_ascend_is_aic() -> None:
        asc.ascend_is_aic()

    kernel_ascend_is_aic[1]()
    assert mock_launcher_run.call_count == 1


def test_ascend_is_aiv(mock_launcher_run):

    @asc.jit
    def kernel_ascend_is_aiv() -> None:
        asc.ascend_is_aiv()

    kernel_ascend_is_aiv[1]()
    assert mock_launcher_run.call_count == 1


def test_pipe_barrier(mock_launcher_run):

    @asc.jit
    def kernel_pipe_barrier() -> None:
        asc.pipe_barrier(0)

    kernel_pipe_barrier[1]()
    assert mock_launcher_run.call_count == 1


def test_reset_mask(mock_launcher_run):

    @asc.jit
    def kernel_reset_mask() -> None:
        asc.reset_mask()

    kernel_reset_mask[1]()
    assert mock_launcher_run.call_count == 1


def test_set_sys_workspace(mock_launcher_run):

    @asc.jit
    def kernel_set_sys_workspace(x: asc.GlobalAddress) -> None:
        asc.set_sys_workspace(x)

    x = MockTensor(asc.uint8)
    kernel_set_sys_workspace[1](x)
    assert mock_launcher_run.call_count == 1


def test_set_wait_flag(mock_launcher_run):

    @asc.jit
    def kernel_set_wait_flag() -> None:
        asc.set_flag(event=asc.HardEvent.V_MTE3, event_id=0)
        asc.wait_flag(event=asc.HardEvent.V_MTE3, event_id=0)

    kernel_set_wait_flag[1]()
    assert mock_launcher_run.call_count == 1


def test_ib_set(mock_launcher_run):

    @asc.jit
    def kernel_ib_set(x: asc.GlobalAddress) -> None:
        gm = asc.GlobalTensor()
        gm.set_global_buffer(x)
        ub = asc.LocalTensor(dtype=asc.int32, pos=asc.TPosition.VECIN, addr=0, tile_size=32)
        asc.ib_set(gm, ub, block_idx=0, event_id=0)

    x = MockTensor(asc.int32)
    kernel_ib_set[1](x)
    assert mock_launcher_run.call_count == 1


def test_ib_wait(mock_launcher_run):

    @asc.jit
    def kernel_ib_wait(x: asc.GlobalAddress) -> None:
        gm = asc.GlobalTensor()
        gm.set_global_buffer(x)
        ub = asc.LocalTensor(dtype=asc.int32, pos=asc.TPosition.VECIN, addr=0, tile_size=32)
        asc.ib_wait(gm, ub, block_idx=0, event_id=0)

    x = MockTensor(asc.int32)
    kernel_ib_wait[1](x)
    assert mock_launcher_run.call_count == 1


def test_sync_all_soft(mock_launcher_run):

    @asc.jit
    def kernel_sync_all_soft(x: asc.GlobalAddress) -> None:
        gm = asc.GlobalTensor()
        gm.set_global_buffer(x)
        ub = asc.LocalTensor(dtype=asc.int32, pos=asc.TPosition.VECIN, addr=0, tile_size=32)
        asc.sync_all(gm, ub, used_cores=0)

    x = MockTensor(asc.int32)
    kernel_sync_all_soft[1](x)
    assert mock_launcher_run.call_count == 1


def test_sync_all_hard(mock_launcher_run):

    @asc.jit
    def kernel_sync_all_hard() -> None:
        asc.sync_all()

    kernel_sync_all_hard[1]()
    assert mock_launcher_run.call_count == 1


def test_cross_core_set_flag(mock_launcher_run):

    @asc.jit
    def kernel_cross_core_set_flag() -> None:
        asc.cross_core_set_flag(flag_id=0, mode_id=0, pipe=asc.PipeID.PIPE_V)

    kernel_cross_core_set_flag[1]()
    assert mock_launcher_run.call_count == 1


def test_cross_core_wait_flag(mock_launcher_run):

    @asc.jit
    def kernel_cross_core_wait_flag() -> None:
        asc.cross_core_wait_flag(flag_id=0, mode_id=0, pipe=asc.PipeID.PIPE_V)

    kernel_cross_core_wait_flag[1]()
    assert mock_launcher_run.call_count == 1


def test_printf(mock_launcher_run):

    @asc.jit
    def kernel_printf(x: asc.GlobalAddress) -> None:
        asc.printf("%s", x)

    x = MockTensor(asc.float16)
    kernel_printf[1](x)
    assert mock_launcher_run.call_count == 1


def test_dump_tensor(mock_launcher_run):

    @asc.jit
    def kernel_dump_tensor(x: asc.GlobalAddress) -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        x_gm = asc.GlobalTensor()
        x_gm.set_global_buffer(x)
        asc.dump_tensor(tensor=x_local, desc=0, dump_size=5)
        asc.dump_tensor(tensor=x_gm, desc=0, dump_size=5)

    x = MockTensor(asc.float16)
    kernel_dump_tensor[1](x)
    assert mock_launcher_run.call_count == 1


def test_device_print(mock_launcher_run):

    @asc.jit
    def kernel_device_print(x: asc.GlobalAddress) -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        x_gm = asc.GlobalTensor()
        x_gm.set_global_buffer(x)
        print("local:", x_local, "global:", x_gm, "index:", asc.get_block_idx(), sep=", ", end=";\n")

    x = MockTensor(asc.float16)
    kernel_device_print[1](x)
    assert mock_launcher_run.call_count == 1


def test_static_print(capsys):
    asc.static_print("hello", "world", sep=", ", end=";\n")
    assert capsys.readouterr().out == "hello, world;\n"


def test_device_assert(mock_launcher_run):

    @asc.jit
    def kernel_device_assert(index: int) -> None:
        asc.device_assert(index >= 0, "index must be non-negative")

    kernel_device_assert[1](0)
    assert mock_launcher_run.call_count == 1


def test_static_assert():
    asc.static_assert(True, "ok")
    with pytest.raises(AssertionError, match="condition failed"):
        asc.static_assert(False, "condition failed")


def test_metrics_prof_start(mock_launcher_run):

    @asc.jit
    def kernel_metrics_prof_start() -> None:
        asc.metrics_prof_start()

    kernel_metrics_prof_start[1]()
    assert mock_launcher_run.call_count == 1


def test_metrics_prof_stop(mock_launcher_run):

    @asc.jit
    def kernel_metrics_prof_stop() -> None:
        asc.metrics_prof_stop()

    kernel_metrics_prof_stop[1]()
    assert mock_launcher_run.call_count == 1


def test_print_time_stamp(mock_launcher_run):

    @asc.jit
    def kernel_print_time_stamp(desc_id: int):
        asc.print_time_stamp(desc_id)

    kernel_print_time_stamp[1](1)
    assert mock_launcher_run.call_count == 1


def test_set_deq_scale(mock_launcher_run):

    @asc.jit
    def kernel_set_deq_scale() -> None:
        asc.set_deq_scale(1.0)
        asc.set_deq_scale(1.0, 5, False)
        vdeq_local = asc.LocalTensor(dtype=asc.uint64, pos=asc.TPosition.VECIN, addr=0, tile_size=16)
        vdeq_scale = [1.0] * 16
        vdeq_offset = [5] * 16
        vdeq_sign_mode = [False] * 16
        vdeq_info = asc.VdeqInfo(vdeq_scale, vdeq_offset, vdeq_sign_mode)
        asc.set_deq_scale(vdeq_local, vdeq_info)

    kernel_set_deq_scale[1]()
    assert mock_launcher_run.call_count == 1


def test_get_program_counter(mock_launcher_run):

    @asc.jit
    def kernel_get_program_counter() -> None:
        pc = asc.get_program_counter()

    kernel_get_program_counter[1]()
    assert mock_launcher_run.call_count == 1


def test_get_system_cycle(mock_launcher_run):

    @asc.jit
    def kernel_get_system_cycle() -> None:
        cycle = asc.get_system_cycle()

    kernel_get_system_cycle[1]()
    assert mock_launcher_run.call_count == 1


def test_trap(mock_launcher_run):

    @asc.jit
    def kernel_trap() -> None:
        asc.trap()

    kernel_trap[1]()


def test_set_hccl_context(mock_launcher_run):

    @asc.jit
    def kernel_set_hccl_context(x: asc.GlobalAddress) -> None:
        asc.set_hccl_context(0, x)

    x = MockTensor(asc.uint8)
    kernel_set_hccl_context[1](x)
    assert mock_launcher_run.call_count == 1


def test_get_hccl_context(mock_launcher_run):

    @asc.jit
    def kernel_get_hccl_context() -> None:
        ctx = asc.get_hccl_context(1)

    kernel_get_hccl_context[1]()
    assert mock_launcher_run.call_count == 1


def test_data_cache_clean_and_invalid(mock_launcher_run):

    @asc.jit
    def kernel_data_cache_clean_and_invalid(dst: asc.GlobalAddress) -> None:
        dst_gm = asc.GlobalTensor()
        dst_gm.set_global_buffer(dst)
        asc.data_cache_clean_and_invalid(entire_type=asc.CacheLine.SINGLE_CACHE_LINE,
                                         dcci_dst=asc.DcciDst.CACHELINE_OUT, dst=dst_gm)
        asc.data_cache_clean_and_invalid(entire_type=asc.CacheLine.SINGLE_CACHE_LINE, dst=dst_gm)

    dst = MockTensor(asc.float32)
    kernel_data_cache_clean_and_invalid[1](dst)
    assert mock_launcher_run.call_count == 1


def test_data_cache_preload(mock_launcher_run):

    @asc.jit
    def kernel_data_cache_preload(src_addr: asc.GlobalAddress) -> None:
        src_gm = asc.GlobalTensor()
        src_gm.set_global_buffer(src_addr)
        asc.data_cache_preload(src=src_gm, cache_offset=1024)

    src_mock = MockTensor(asc.uint64)
    kernel_data_cache_preload[1](src_mock)
    assert mock_launcher_run.call_count == 1


def test_get_icache_preload_status(mock_launcher_run):

    @asc.jit
    def kernel_get_icache_preload_status() -> None:
        cache_preload_status = asc.get_icache_preload_status()

    kernel_get_icache_preload_status[1]()
    assert mock_launcher_run.call_count == 1


def test_icache_preload(mock_launcher_run):

    @asc.jit
    def kernel_icache_preload() -> None:
        pre_fetch_len = 2
        asc.icache_preload(pre_fetch_len)

    kernel_icache_preload[1]()
    assert mock_launcher_run.call_count == 1


def test_set_hf32_mode(mock_launcher_run):

    @asc.jit
    def kernel_set_hf32_mode() -> None:
        asc.set_hf32_mode(True)
        asc.set_hf32_mode(False)

    kernel_set_hf32_mode[1]()
    assert mock_launcher_run.call_count == 1


def test_set_hf32_trans_mode(mock_launcher_run):

    @asc.jit
    def kernel_set_hf32_trans_mode() -> None:
        asc.set_hf32_trans_mode(True)
        asc.set_hf32_trans_mode(False)

    kernel_set_hf32_trans_mode[1]()
    assert mock_launcher_run.call_count == 1


def test_set_mm_layout_transform(mock_launcher_run):

    @asc.jit
    def kernel_set_mm_layout_transform() -> None:
        asc.set_mm_layout_transform(True)
        asc.set_mm_layout_transform(False)

    kernel_set_mm_layout_transform[1]()
    assert mock_launcher_run.call_count == 1


def test_set_mask_count(mock_launcher_run):

    @asc.jit
    def kernel_set_mask_count() -> None:
        asc.set_mask_count()

    kernel_set_mask_count[1]()
    assert mock_launcher_run.call_count == 1


def test_set_mask_norm(mock_launcher_run):

    @asc.jit
    def kernel_set_mask_norm() -> None:
        asc.set_mask_norm()

    kernel_set_mask_norm[1]()
    assert mock_launcher_run.call_count == 1


def test_dump_acc_chk_point(mock_launcher_run):

    @asc.jit
    def kernel_dump_acc_chk_point(x: asc.GlobalAddress) -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        x_gm = asc.GlobalTensor()
        x_gm.set_global_buffer(x)
        asc.dump_acc_chk_point(tensor=x_local, index=0, count_off=1, dump_size=5)
        asc.dump_acc_chk_point(tensor=x_gm, index=0, count_off=1, dump_size=5)

    x = MockTensor(asc.float16)
    kernel_dump_acc_chk_point[1](x)
    assert mock_launcher_run.call_count == 1


def test_proposal_concat(mock_launcher_run):

    @asc.jit
    def kernel_proposal_concat() -> None:
        dst = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=256)
        src = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=256)
        asc.proposal_concat(dst, src, repeat_time=2, mode_number=4)

    kernel_proposal_concat[1]()
    assert mock_launcher_run.call_count == 1


def test_proposal_extract(mock_launcher_run):

    @asc.jit
    def kernel_proposal_extract() -> None:
        dst = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=256)
        src = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=256)
        asc.proposal_extract(dst, src, repeat_time=2, mode_number=4)

    kernel_proposal_extract[1]()
    assert mock_launcher_run.call_count == 1


def test_set_vector_mask_kernel(mock_launcher_run):

    @asc.jit
    def set_vector_mask_kernel():
        asc.set_vector_mask(128, dtype=asc.float16, mode=asc.MaskMode.COUNTER)
        uint64_max = 2**64 - 1
        asc.set_vector_mask(uint64_max, uint64_max, dtype=asc.float16, mode=asc.MaskMode.NORMAL)

    set_vector_mask_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_get_mrg_sort_result(mock_launcher_run):

    @asc.jit
    def kernel_get_mrg_sort_result():
        mrg1, mrg2, mrg3, mrg4 = asc.get_mrg_sort_result()

    result = kernel_get_mrg_sort_result[1]()
    assert mock_launcher_run.call_count == 1


def test_mrg_sort_kernel(mock_launcher_run):

    @asc.jit
    def mrg_sort_kernel():
        src1 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        src2 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=512, tile_size=512)
        src3 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=1024, tile_size=512)
        src4 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=1536, tile_size=512)
        dst = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=2048)
        sort_list = asc.MrgSortSrcList(asc.float16, src1, src2, src3, src4)
        element_count_list = [128, 128, 128, 128]
        sorted_num = [0, 0, 0, 0]
        asc.mrg_sort(dst, sort_list, element_count_list, sorted_num, valid_bit=15, repeat_time=1)
        asc.mrg_sort(dst, sort_list, element_count_list, sorted_num, valid_bit=15, repeat_time=1,
                     is_exhausted_suspension=True)
        mrg_sort4_info = asc.MrgSort4Info(element_count_list, if_exhausted_suspension=False, valid_bit=7,
                                          repeat_times=1)
        asc.mrg_sort(dst, sort_list, mrg_sort4_info)

    mrg_sort_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_mrg_sort4_kernel(mock_launcher_run):

    @asc.jit
    def mrg_sort4_kernel():
        src1 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        src2 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=512, tile_size=512)
        src3 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=1024, tile_size=512)
        src4 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=1536, tile_size=512)
        dst = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=2048)
        src_list = asc.MrgSortSrcList(asc.float16, src1, src2, src3, src4)
        element_lengths = [16, 16, 16, 16]
        mrg_sort4_info = asc.MrgSort4Info(element_lengths, if_exhausted_suspension=False, valid_bit=7, repeat_times=1)
        asc.mrg_sort4(dst, src_list, mrg_sort4_info)

    mrg_sort4_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_rp_sort16_kernel(mock_launcher_run):

    @asc.jit
    def rp_sort16_kernel():
        src = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        dst = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.rp_sort16(dst, src, repeat_time=2)

    rp_sort16_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_sort32_kernel(mock_launcher_run):

    @asc.jit
    def sort32_kernel():
        src0 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        src1 = asc.LocalTensor(dtype=asc.uint32, pos=asc.TPosition.VECIN, addr=512, tile_size=512)
        dst = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=1024)
        asc.sort32(dst, src0, src1, repeat_time=4)

    sort32_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_sort_kernel(mock_launcher_run):

    @asc.jit
    def sort_kernel():
        concat = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        index = asc.LocalTensor(dtype=asc.uint32, pos=asc.TPosition.VECIN, addr=512, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=1024, tile_size=512)
        dst = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=1024)
        asc.sort(dst, concat, index, tmp, repeat_time=4)

    sort_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_transpose_kernel(mock_launcher_run):

    @asc.jit
    def transpose_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.transpose(z_local, x_local)
        params = asc.TransposeParamsExt(n_size=1, c_size=16, h_size=4, w_size=4,
                                        transpose_type=asc.TransposeType.TRANSPOSE_NCHW2NHWC)
        tmp_buffer = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.transpose(z_local, x_local, tmp_buffer, params)

    transpose_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_trans_data_to_5hd_kernel(mock_launcher_run):

    @asc.jit
    def trans_data_to_5hd_kernel():
        x = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        x_local = asc.LocalTensor(dtype=asc.uint64)
        z_local = asc.LocalTensor(dtype=asc.uint64)
        src_addr = x[0].get_phy_addr()
        x_local.set_value(0, src_addr)
        dst_addr = z[0].get_phy_addr()
        z_local.set_value(0, dst_addr)
        params = asc.TransDataTo5HDParams(dst_high_half=False, src_high_half=False, repeat_times=16, dst_rep_stride=16,
                                          src_rep_stride=1)
        asc.trans_data_to_5hd(z_local, x_local, params)
        dst_list = [
            z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9], z[10], z[11], z[12], z[13], z[14], z[15]
        ]
        src_list = [
            x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15]
        ]
        asc.trans_data_to_5hd(dst_list, src_list, params)
        addr_dst_list = [
            dst_addr, dst_addr, dst_addr, dst_addr, dst_addr, dst_addr, dst_addr, dst_addr, dst_addr, dst_addr,
            dst_addr, dst_addr, dst_addr, dst_addr, dst_addr, dst_addr
        ]
        addr_src_list = [
            src_addr, src_addr, src_addr, src_addr, src_addr, src_addr, src_addr, src_addr, src_addr, src_addr,
            src_addr, src_addr, src_addr, src_addr, src_addr, src_addr
        ]
        asc.trans_data_to_5hd(addr_dst_list, addr_src_list, params)

    trans_data_to_5hd_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_init_soc_state(mock_launcher_run):

    @asc.jit
    def kernel_init_soc_state() -> None:
        asc.init_soc_state()

    kernel_init_soc_state[1]()
    assert mock_launcher_run.call_count == 1


def test_set_store_atomic_config(mock_launcher_run):

    @asc.jit
    def kernel_set_store_atomic_config() -> None:
        asc.set_store_atomic_config(asc.AtomicDtype.ATOMIC_F16, asc.AtomicOp.ATOMIC_SUM)

    kernel_set_store_atomic_config[1]()
    assert mock_launcher_run.call_count == 1


def test_get_store_atomic_config(mock_launcher_run):

    @asc.jit
    def kernel_get_store_atomic_config() -> None:
        asc.set_store_atomic_config(asc.AtomicDtype.ATOMIC_F16, asc.AtomicOp.ATOMIC_SUM)
        atomic_type, atomic_op = asc.get_store_atomic_config()

    kernel_get_store_atomic_config[1]()
    assert mock_launcher_run.call_count == 1


def test_check_local_memory_ia(mock_launcher_run):

    @asc.jit
    def kernel_check_local_memory_ia() -> None:
        params = asc.CheckLocalMemoryIAParam()
        asc.check_local_memory_ia(params)

    kernel_check_local_memory_ia[1]()
    assert mock_launcher_run.call_count == 1
