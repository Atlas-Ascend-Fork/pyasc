# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest

import asc
from asc.runtime import config
from asc.runtime.jit import MockTensor


def setup_function():
    config.set_platform(config.Backend.Model, check=False)


def test_init(mock_launcher_run):

    @asc.jit
    def kernel_init(workspace: asc.GlobalAddress) -> None:
        pipe = asc.TPipe()
        a = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        b = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        c = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        matmul = asc.adv.Matmul(a, b, c)
        asc.adv.register_matmul(pipe, workspace, matmul)

    workspace = MockTensor(asc.uint8)
    kernel_init[1](workspace)
    assert mock_launcher_run.call_count == 1


def test_end(mock_launcher_run):

    @asc.jit
    def kernel_end(workspace: asc.GlobalAddress) -> None:
        pipe = asc.TPipe()
        a = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        b = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        c = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        matmul = asc.adv.Matmul(a, b, c)
        asc.adv.register_matmul(pipe, workspace, matmul)
        matmul.end()

    workspace = MockTensor(asc.uint8)
    kernel_end[1](workspace)
    assert mock_launcher_run.call_count == 1


def test_get_tensor_c(mock_launcher_run):

    @asc.jit
    def kernel_get_tensor_c(c_addr: asc.GlobalAddress, workspace: asc.GlobalAddress) -> None:
        pipe = asc.TPipe()
        a = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        b = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        c = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.NZ, dtype=asc.float16)
        matmul = asc.adv.Matmul(a, b, c)
        asc.adv.register_matmul(pipe, workspace, matmul)
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        matmul.get_tensor_c(tensor=x_local)
        c_global = asc.GlobalTensor()
        c_global.set_global_buffer(c_addr)
        matmul.get_tensor_c(tensor=c_global, optional_tensor=x_local)
        matmul.get_batch_tensor_c(x_local, 2, 2, sync=False)
        matmul.get_batch_tensor_c(2, 2, False, False)

    c = MockTensor(asc.float16)
    workspace = MockTensor(asc.uint8)
    kernel_get_tensor_c[1](c, workspace)
    assert mock_launcher_run.call_count == 1


def test_initialize(mock_launcher_run):

    @asc.jit
    def kernel_initialize(workspace: asc.GlobalAddress) -> None:
        pipe = asc.TPipe()
        a = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        b = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        c = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        matmul = asc.adv.Matmul(a, b, c)
        asc.adv.register_matmul(pipe, workspace, matmul)
        tiling = asc.adv.TCubeTiling(used_core_num=24, m=512, k_a=512, k_b=512, n=512, base_m=64, base_k=64, base_n=64,
                                     single_core_m=256, single_core_k=256, single_core_n=256, depth_a1=1, depth_b1=1,
                                     step_m=1, step_n=1, share_mode=0, share_ub_size=0,
                                     share_l1_size=asc.property(asc.TOTAL_L1_SIZE),
                                     share_l0c_size=asc.property(asc.TOTAL_L0C_SIZE))
        matmul.init(tiling)

    workspace = MockTensor(asc.uint8)
    kernel_initialize[1](workspace)
    assert mock_launcher_run.call_count == 1


def test_set_self_define_data(mock_launcher_run):

    @asc.jit
    def kernel_set_self_define_data(a: asc.GlobalAddress, workspace: asc.GlobalAddress):
        pipe = asc.TPipe()
        a_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        b_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        c_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        matmul = asc.adv.Matmul(a_type, b_type, c_type)
        asc.adv.register_matmul(pipe, workspace, matmul)
        tiling = asc.adv.TCubeTiling(used_core_num=24, m=512, k_a=512, k_b=512, n=512, base_m=64, base_k=64, base_n=64,
                                     single_core_m=256, single_core_k=256, single_core_n=256, depth_a1=1, depth_b1=1,
                                     step_m=1, step_n=1, share_mode=0, share_ub_size=0,
                                     share_l1_size=asc.property(asc.TOTAL_L1_SIZE),
                                     share_l0c_size=asc.property(asc.TOTAL_L0C_SIZE))
        matmul.init(tiling)
        matmul.set_self_define_data(a)

    a = MockTensor(asc.float16)
    workspace = MockTensor(asc.uint8)
    kernel_set_self_define_data[1](a, workspace)
    assert mock_launcher_run.call_count == 1


def test_set_user_def_info(mock_launcher_run):

    @asc.jit
    def kernel_set_user_def_info(a: asc.GlobalAddress, workspace: asc.GlobalAddress):
        pipe = asc.TPipe()
        a_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        b_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        c_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        matmul = asc.adv.Matmul(a_type, b_type, c_type)
        asc.adv.register_matmul(pipe, workspace, matmul)
        tiling = asc.adv.TCubeTiling(used_core_num=24, m=512, k_a=512, k_b=512, n=512, base_m=64, base_k=64, base_n=64,
                                     single_core_m=256, single_core_k=256, single_core_n=256, depth_a1=1, depth_b1=1,
                                     step_m=1, step_n=1, share_mode=0, share_ub_size=0,
                                     share_l1_size=asc.property(asc.TOTAL_L1_SIZE),
                                     share_l0c_size=asc.property(asc.TOTAL_L0C_SIZE))
        matmul.init(tiling)
        matmul.set_user_def_info(a)

    a = MockTensor(asc.float16)
    workspace = MockTensor(asc.uint8)
    kernel_set_user_def_info[1](a, workspace)
    assert mock_launcher_run.call_count == 1


def test_get_matmul_api_tiling(mock_launcher_run):

    @asc.jit
    def kernel_get_matmul_api_tiling(workspace: asc.GlobalAddress):
        pipe = asc.TPipe()
        a_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        b_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        c_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        matmul = asc.adv.Matmul(a_type, b_type, c_type)
        asc.adv.register_matmul(pipe, workspace, matmul)

        mm_cfg = asc.adv.MatmulConfig(do_norm=False, iterate_mode=asc.IterateMode.ITERATE_MODE_DEFAULT,
                                      single_core_mn=0, schedule_type=asc.ScheduleType.INNER_PRODUCT)
        matmul_api_static = asc.adv.get_matmul_api_tiling(mm_cfg=mm_cfg, l1_size=524288, a_type=a_type, b_type=b_type,
                                                          c_type=c_type, bias_type=c_type)
        asc.adv.get_matmul_api_tiling(mm_cfg, 524288, a_type, b_type, c_type, c_type)
        asc.adv.get_matmul_api_tiling(mm_cfg, 524288, a_type, b_type, c_type)

    workspace = MockTensor(asc.uint8)
    kernel_get_matmul_api_tiling[1](workspace)
    assert mock_launcher_run.call_count == 1


def test_iterate(mock_launcher_run):

    @asc.jit
    def kernel_iterate(a: asc.GlobalAddress, b: asc.GlobalAddress, c: asc.GlobalAddress,
                       workspace: asc.GlobalAddress) -> None:
        pipe = asc.TPipe()
        a_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        b_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        c_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        matmul = asc.adv.Matmul(a_type, b_type, c_type)
        asc.adv.register_matmul(pipe, workspace, matmul)
        tiling = asc.adv.TCubeTiling(used_core_num=24, m=512, k_a=512, k_b=512, n=512, base_m=64, base_k=64, base_n=64,
                                     single_core_m=256, single_core_k=256, single_core_n=256, depth_a1=1, depth_b1=1,
                                     step_m=1, step_n=1, share_mode=0, share_ub_size=0,
                                     share_l1_size=asc.property(asc.TOTAL_L1_SIZE),
                                     share_l0c_size=asc.property(asc.TOTAL_L0C_SIZE))
        matmul.init(tiling)
        a_global = asc.GlobalTensor()
        b_global = asc.GlobalTensor()
        c_global = asc.GlobalTensor()
        a_global.set_global_buffer(a)
        b_global.set_global_buffer(b)
        c_global.set_global_buffer(c)
        matmul.set_tensor_a(a_global)
        matmul.set_tensor_b(b_global)
        matmul.set_tensor_a(8)
        matmul.set_tensor_b(8)
        x_local = asc.LocalTensor(dtype=asc.float32, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        bias_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        matmul.set_bias(bias_local)
        matmul.disable_bias()
        with matmul.iterate() as count:
            ...
        matmul.end()

    a = MockTensor(asc.float16)
    b = MockTensor(asc.float16)
    c = MockTensor(asc.float16)
    workspace = MockTensor(asc.uint8)
    kernel_iterate[1](a, b, c, workspace)
    assert mock_launcher_run.call_count == 1


def test_iterate_all(mock_launcher_run):

    @asc.jit
    def kernel_iterate_all(a: asc.GlobalAddress, b: asc.GlobalAddress, c: asc.GlobalAddress,
                           workspace: asc.GlobalAddress, quant_vector: asc.GlobalAddress) -> None:
        pipe = asc.TPipe()
        a_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        b_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        c_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        matmul = asc.adv.Matmul(a_type, b_type, c_type)
        asc.adv.register_matmul(pipe, workspace, matmul)
        m, n, k_a, k_b, k_c = 512, 512, 512, 512, 512
        single_core_m, single_core_n, single_core_k = 256, 256, 256
        base_m, base_n, base_k = 128, 256, 64
        tiling = asc.adv.TCubeTiling(used_core_num=24, m=m, k_a=k_a, k_b=k_b, n=n, \
            base_m=base_m, base_k=base_k, base_n=base_n, \
            single_core_m=single_core_m, single_core_k=single_core_k, single_core_n=single_core_n, \
            depth_a1=1, depth_b1=1, step_m=1, step_n=1, share_mode=0, share_ub_size=0, \
            share_l1_size=asc.property(asc.TOTAL_L1_SIZE), share_l0c_size=asc.property(asc.TOTAL_L0C_SIZE))
        matmul.init(tiling)
        a_global = asc.GlobalTensor()
        b_global = asc.GlobalTensor()
        c_global = asc.GlobalTensor()
        quant_global = asc.GlobalTensor()
        a_global.set_global_buffer(a)
        b_global.set_global_buffer(b)
        c_global.set_global_buffer(c)
        quant_global.set_global_buffer(quant_vector)
        matmul.set_tensor_a(a_global)
        matmul.set_tensor_b(b_global)
        matmul.set_quant_scalar(2)
        matmul.set_quant_vector(quant_global)
        matmul.set_org_shape(m, n, k_a)
        matmul.set_org_shape(m, n, k_a, k_b)
        matmul.set_org_shape(m, n, k_a, k_b, k_c)
        matmul.set_single_shape(single_core_m, single_core_n, single_core_k)
        matmul.iterate_all(c_global, en_atomic=0, en_sequential_write=False, wait_iterate_all=True, fake_msg=False,
                           sync=False)
        matmul.wait_iterate_all()
        ub_matrix = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        matmul.iterate_all(ub_matrix)
        matmul.end()

    a = MockTensor(asc.float16)
    b = MockTensor(asc.float16)
    c = MockTensor(asc.float16)
    workspace = MockTensor(asc.uint8)
    quant_vector = MockTensor(asc.uint64)
    kernel_iterate_all[1](a, b, c, workspace, quant_vector)
    assert mock_launcher_run.call_count == 1


def test_iterate_batch(mock_launcher_run):

    @asc.jit
    def kernel_iterate_batch(a: asc.GlobalAddress, b: asc.GlobalAddress, c: asc.GlobalAddress,
                             workspace: asc.GlobalAddress) -> None:
        pipe = asc.TPipe()
        a_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        b_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        c_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        matmul = asc.adv.Matmul(a_type, b_type, c_type)
        asc.adv.register_matmul(pipe, workspace, matmul)
        tiling = asc.adv.TCubeTiling(used_core_num=24, m=512, k_a=512, k_b=512, n=512, base_m=64, base_k=64, base_n=64,
                                     single_core_m=256, single_core_k=256, single_core_n=256, depth_a1=1, depth_b1=1,
                                     step_m=1, step_n=1, share_mode=0, share_ub_size=0,
                                     share_l1_size=asc.property(asc.TOTAL_L1_SIZE),
                                     share_l0c_size=asc.property(asc.TOTAL_L0C_SIZE))
        matmul.init(tiling)
        a_global = asc.GlobalTensor()
        b_global = asc.GlobalTensor()
        c_global = asc.GlobalTensor()
        a_global.set_global_buffer(a)
        b_global.set_global_buffer(b)
        c_global.set_global_buffer(c)
        matmul.set_tensor_a(a_global)
        matmul.set_tensor_b(b_global)
        matmul.iterate_batch(c_global, 2, 2, en_sequential_write=False, matrix_stride_a=0, matrix_stride_b=0,
                             matrix_stride_c=0, sync=False, wait_iterate_batch=True)
        matmul.wait_iterate_batch()
        matmul.end()

    a = MockTensor(asc.float16)
    b = MockTensor(asc.float16)
    c = MockTensor(asc.float16)
    workspace = MockTensor(asc.uint8)
    kernel_iterate_batch[1](a, b, c, workspace)
    assert mock_launcher_run.call_count == 1


def test_iterate_n_batch(mock_launcher_run):

    @asc.jit
    def kernel_iterate_n_batch(a: asc.GlobalAddress, b: asc.GlobalAddress, c: asc.GlobalAddress,
                               workspace: asc.GlobalAddress) -> None:
        pipe = asc.TPipe()
        a_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16,
                                    is_trans=False, layout=asc.LayoutMode.BSNGD)
        b_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16,
                                    is_trans=True, layout=asc.LayoutMode.BSNGD)
        c_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16,
                                    layout=asc.LayoutMode.BSNGD)
        matmul = asc.adv.Matmul(a_type, b_type, c_type)
        asc.adv.register_matmul(pipe, workspace, matmul)
        tiling = asc.adv.TCubeTiling(used_core_num=24, m=512, k_a=512, k_b=512, n=512, base_m=64, base_k=64, base_n=64,
                                     single_core_m=256, single_core_k=256, single_core_n=256, depth_a1=1, depth_b1=1,
                                     step_m=1, step_n=1, share_mode=0, share_ub_size=0,
                                     share_l1_size=asc.property(asc.TOTAL_L1_SIZE),
                                     share_l0c_size=asc.property(asc.TOTAL_L0C_SIZE))
        matmul.init(tiling)
        a_global = asc.GlobalTensor()
        b_global = asc.GlobalTensor()
        c_global = asc.GlobalTensor()
        a_global.set_global_buffer(a)
        b_global.set_global_buffer(b)
        c_global.set_global_buffer(c)
        matmul.set_tensor_a(a_global)
        matmul.set_tensor_b(b_global)
        matmul.set_hf32(True)
        matmul.set_hf32(True, False)
        matmul.set_tail(31, 15, 7)
        matmul.iterate_n_batch(64, 64, 64, False)
        matmul.iterate_n_batch(64, 64, 64, False, 1, 1)
        matmul.iterate_n_batch(64, 64, 64, False, 1, 1, 1, False, True)
        matmul.set_workspace(workspace, 1024)
        workspace_global = asc.GlobalTensor()
        workspace_global.set_global_buffer(workspace)
        matmul.set_workspace(workspace_global)
        matmul.wait_get_tensor_c()
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        matmul.async_get_tensor_c(c=x_local)
        matmul.end()

    a = MockTensor(asc.float16)
    b = MockTensor(asc.float16)
    c = MockTensor(asc.float16)
    workspace = MockTensor(asc.uint8)
    kernel_iterate_n_batch[1](a, b, c, workspace)
    assert mock_launcher_run.call_count == 1


def test_get_config(mock_launcher_run):

    @asc.jit
    def kernel_get_config(workspace: asc.GlobalAddress):
        pipe = asc.TPipe()
        a_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        b_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        c_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16)
        matmul = asc.adv.Matmul(a_type, b_type, c_type)
        asc.adv.register_matmul(pipe, workspace, matmul)
        config_mode = asc.MatmulConfigMode.CONFIG_NORM
        shape_params = asc.adv.MatmulShapeParams(128, 128, 128, 64, 64, 64)
        matmul_quant_params = asc.adv.MatmulQuantParams(False, False)
        matmul_batch_params = asc.adv.MatmulBatchParams(False)
        matmul_func_params = asc.adv.MatmulFuncParams(False, True)
        asc.adv.QuantConfig(512, 64, 2, 1024)
        asc.adv.get_mm_config(config_mode, shape_params, matmul_quant_params, matmul_batch_params, matmul_func_params)
        asc.adv.get_special_basic_config(256, 256, 256, 16, 16, 16, 2, 2)
        asc.adv.get_normal_config()
        asc.adv.get_mdl_config()
        asc.adv.get_special_mdl_config()
        asc.adv.get_ib_share_norm_config()
        asc.adv.get_mm_config()

    workspace = MockTensor(asc.uint8)
    kernel_get_config[1](workspace)
    assert mock_launcher_run.call_count == 1


def test_cube_only(mock_launcher_run):

    @asc.jit(matmul_cube_only=True)
    def kernel_cube_only(x_gm: asc.GlobalAddress, workspace: asc.GlobalAddress):
        pipe = asc.TPipe()
        a_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16,
                                    is_trans=False, layout=asc.LayoutMode.BSNGD)
        b_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16,
                                    is_trans=False, layout=asc.LayoutMode.BSNGD)
        c_type = asc.adv.MatmulType(position=asc.TPosition.GM, format=asc.CubeFormat.ND, dtype=asc.float16,
                                    is_trans=False, layout=asc.LayoutMode.BSNGD)
        matmul = asc.adv.Matmul(a_type, b_type, c_type)
        asc.adv.register_matmul(pipe, workspace, matmul)
        x = asc.GlobalTensor()
        x.set_global_buffer(x_gm)
        matmul.set_sparse_index(x)
        matmul.set_batch_num(2, 2)

    x = MockTensor(asc.uint8)
    workspace = MockTensor(asc.uint8)
    kernel_cube_only[1](x, workspace)
    assert mock_launcher_run.call_count == 1


@pytest.mark.parametrize("api,args", [("get_tensor_c", ()), ("get_batch_tensor_c", (2, 2))])
def test_matmul_get_result_returns_dispatched_tensor(monkeypatch, api, args):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from asc.language.adv import matmul as matmul_module

    ir_result = object()
    tensor_result = object()
    builder = SimpleNamespace(
        create_asc_MatmulGetTensorCReturnOp=Mock(return_value=ir_result),
        create_asc_MatmulGetBatchTensorCOp=Mock(return_value=ir_result),
    )
    monkeypatch.setattr(matmul_module.global_builder, "get_ir_builder", lambda: builder)
    monkeypatch.setattr(matmul_module.ir, "get_global_tensor_type", lambda dtype: dtype)
    monkeypatch.setattr(matmul_module, "_mat", lambda value, dtype: SimpleNamespace(to_ir=lambda: value))
    wrap_tensor = Mock(return_value=tensor_result)
    monkeypatch.setattr(matmul_module, "GlobalTensor", wrap_tensor)
    owner = matmul_module.Matmul(handle=object())
    owner.c_dtype = SimpleNamespace(to_ir=lambda: object())

    # Exercise the Python dispatcher without the outer JIT initialization guard.
    result = getattr(matmul_module.Matmul, api).__wrapped__(owner, *args, sync=False)
    assert result is tensor_result
    wrap_tensor.assert_called_once_with(ir_result)
