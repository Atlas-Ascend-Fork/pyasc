# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from asc.experimental import asctile
import pytest
import torch


@asctile.jit(cv_ratio=2, reuse_alloc=2, vf_fusion=True)
def flash_attention_pyasc(q_ptr: asctile.GlobalAddress, k_ptr: asctile.GlobalAddress, v_ptr: asctile.GlobalAddress,
                          o_ptr: asctile.GlobalAddress, ws_o_ptr: asctile.GlobalAddress, B: asctile.ConstExpr,
                          S1: asctile.ConstExpr, S2: asctile.ConstExpr, N: asctile.ConstExpr, N_kv: asctile.ConstExpr,
                          D: asctile.ConstExpr, step1: asctile.ConstExpr, step2: asctile.ConstExpr,
                          stepMM1: asctile.ConstExpr, scale_factor: asctile.ConstExpr, g_group: asctile.ConstExpr,
                          d_v: asctile.ConstExpr, dtype: asctile.ConstExpr, s2_unroll: asctile.ConstExpr[int],
                          f_unroll: asctile.ConstExpr[int], f_unroll_norm: asctile.ConstExpr[int]):
    q_gm = asctile.global_tensor(q_ptr, [B * N * S1, D])
    k_gm = asctile.global_tensor(k_ptr, [B * N_kv * S2, D])
    v_gm = asctile.global_tensor(v_ptr, [B * N_kv * S2, D])
    o_gm = asctile.global_tensor(o_ptr, [B * N * S1, D])
    ws_o_gm = asctile.global_tensor(ws_o_ptr, [B * N * S1, D])

    d_mm_steps = asctile.ceildiv(D, stepMM1)
    s2_steps = asctile.ceildiv(S2, step2)
    d_frags = D // d_v

    N_repeats = N // N_kv
    g_chunks = N_repeats // g_group
    s1_batch = g_group * step1
    total_tasks = B * N_kv * g_chunks

    start = asctile.block_idx() / asctile.sub_block_num()
    step = asctile.block_num()

    # One task = one (batch, kv-head, g-chunk), g-chunk minor so blocks running
    # concurrently share the same K/V stream (L2 reuse).
    for task in asctile.range(start, total_tasks, step, unroll_factor=1):
        g_chunk = task % g_chunks
        kv_task = task // g_chunks
        n_kv_local = kv_task % N_kv
        b_idx = kv_task // N_kv
        n_kv = b_idx * N_kv + n_kv_local
        g_start = g_chunk * g_group
        q_row = (b_idx * N + n_kv_local * N_repeats + g_start) * S1

        # Q tile: L1-resident for the whole task, reused by every S2 step.
        q_part = asctile.copy_in(q_gm, [q_row, 0], [s1_batch, D], location=asctile.TensorLocation.L1)

        # Online-softmax row statistics stay UB-resident for the whole task; the O
        # accumulator lives in the GM workspace, zero-initialized on the host (the
        # first S2 step rescales it by exp_max = exp(-1e10 - max) = 0, so no
        # in-kernel init is needed and a stale workspace is harmless too).
        softmax_max = asctile.full([s1_batch], -1e10, dtype=asctile.float32)
        softmax_d = asctile.zeros([s1_batch], dtype=asctile.float32)

        # S2 innermost loop: stream K/V tiles GM->L1. Every step stores the O fragments
        # to the GM workspace and the next step reads them back on the same AIV, so the
        # loop needs gm_barrier (the same-core MTE3 -> MTE2 barrier) around the
        # round-trip. The barrier is inserted once at the end of the (unrolled) loop
        # body, so it is only correct with unroll_factor=1: with unrolling, the store
        # of one step and the load of the next would end up unguarded inside one body.
        for j in asctile.range(0, s2_steps, unroll_factor=s2_unroll, gm_barrier=True):
            s2 = n_kv * S2 + j * step2
            k_part = asctile.copy_in(k_gm, [s2, 0], [step2, D], location=asctile.TensorLocation.L1).transpose()
            mm_acc = asctile.zeros_acc([s1_batch, step2], dtype=asctile.float32)
            for k in asctile.range(d_mm_steps, unroll_factor=2):
                l0q = asctile.copy(q_part, [0, k * stepMM1], [s1_batch, stepMM1], location=asctile.TensorLocation.L0A)
                l0k = asctile.copy(k_part, [k * stepMM1, 0], [stepMM1, step2], location=asctile.TensorLocation.L0B)
                asctile.matmul_acc(mm_acc, l0q, l0k)
            with asctile.cv_strategy(asctile.DistribMode.SplitByM):
                attn_weight = asctile.copy(mm_acc, location=asctile.TensorLocation.UB)
                attn_weight = attn_weight * scale_factor
                max_tmp = asctile.reduce_max(attn_weight, 1)
                x_exp = asctile.exp(attn_weight - max_tmp.reshape(s1_batch, 1).broadcast_to(s1_batch, step2))
                sum_tmp = asctile.reduce_sum(x_exp, 1)
                x_max = asctile.maximum(softmax_max, max_tmp)
                beta = asctile.exp(max_tmp - x_max)
                exp_max = asctile.exp(softmax_max - x_max)
                x_sum = exp_max * softmax_d + beta * sum_tmp
                beta = x_exp * beta.reshape(s1_batch, 1).broadcast_to(s1_batch, step2)
                softmax_max = x_max
                softmax_d = x_sum
            l1 = asctile.copy(beta.to(dtype), location=asctile.TensorLocation.L1)
            l1v = asctile.copy_in(v_gm, [s2, 0], [step2, D], location=asctile.TensorLocation.L1)
            l0b = asctile.copy(l1, [0, 0], [s1_batch, step2], location=asctile.TensorLocation.L0A)
            # BMM2 fragmented over D: [s1_batch, step2] @ [step2, d_v]; the O fragment is
            # read from the GM workspace, rescaled and accumulated, then written back.
            for f in asctile.range(0, d_frags, unroll_factor=f_unroll):
                l0v = asctile.copy(l1v, [0, f * d_v], [step2, d_v], location=asctile.TensorLocation.L0B)
                pv = l0b @ l0v
                with asctile.cv_strategy(asctile.DistribMode.SplitByM):
                    mm_res = asctile.copy(pv, location=asctile.TensorLocation.UB)
                    o_frag = asctile.copy_in(ws_o_gm, [q_row, f * d_v], [s1_batch, d_v],
                                             location=asctile.TensorLocation.UB)
                    exp_max_b = exp_max.reshape(s1_batch, 1).broadcast_to(s1_batch, d_v)
                    o_frag = o_frag * exp_max_b + mm_res
                    asctile.copy_out(o_frag, ws_o_gm, [q_row, f * d_v])

        # Final normalize: read the O accumulator back from GM, one fragment at a time.
        for f in asctile.range(0, d_frags, unroll_factor=f_unroll_norm):
            o_frag = asctile.copy_in(ws_o_gm, [q_row, f * d_v], [s1_batch, d_v], location=asctile.TensorLocation.UB)
            d_b = softmax_d.reshape(s1_batch, 1).broadcast_to(s1_batch, d_v)
            update = o_frag / d_b
            asctile.copy_out(update.to(dtype), o_gm, [q_row, f * d_v])


def flash_attention_torch_reference(q, k, v, B, S1, S2, N, N_kv, D, Br, base_K, scale):
    q_ref = q.to(torch.float32)  # [B, N, S1, D]
    k_ref = k.to(torch.float32).repeat_interleave(N // N_kv, dim=1)  # [B, N_kv, S2, D]
    v_ref = v.to(torch.float32).repeat_interleave(N // N_kv, dim=1)  # [B, N_kv, S2, D]
    o_ref = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, scale=scale, is_causal=False,
                                                             dropout_p=0.0)
    return o_ref


# Q: [B, S1, N, D]
# V,K: [B, S2, N_k, D]
# yapf: disable
test_cases = [
    # PYASC_TESTS_BEGIN
    pytest.param(
        "case-8-1-16-512", 36, ([8, 1, 16, 512], [8, 4096, 1, 512], [8, 4096, 1, 512], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
         [], [], [], [], [8, 1, 16, 64], [8, 4096, 1, 64], [], [], [], [], []),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        ([8, 1, 16, 512], [8, 16, 1, 1]), (torch.bfloat16,),
        (16, 0.041666666666666664, 2147483647, 2147483647, "BSND", 1, 0, 0, 0, 0, False, 0, 0, 0),
        ((), (), (), 16, 0.041666666666666664, 2147483647, 2147483647, "BSND", 1, 0, 0, 0, 0, False, 0, 0, 0),
        132385025, None, 16, 2, 2, 4, id="case-8-1-16-512"),
    pytest.param(
        "case-8-1-128-512", 36, ([8, 1, 128, 512], [8, 4096, 1, 512], [8, 4096, 1, 512], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
         [], [], [], [], [8, 1, 128, 64], [8, 4096, 1, 64], [], [], [], [], []),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        ([8, 1, 128, 512], [8, 128, 1, 1]), (torch.bfloat16,),
        (128, 0.041666666666666664, 2147483647, 2147483647, "BSND", 1, 0, 0, 0, 0, False, 0, 0, 0),
        ((), (), (), 128, 0.041666666666666664, 2147483647, 2147483647, "BSND", 1, 0, 0, 0, 0, False, 0, 0, 0),
        132385025, None, 32, 2, 2, 2, id="case-8-1-128-512"),
    pytest.param(
        "case-64-64-1-512", 36, ([64, 64, 1, 512], [64, 1, 4096, 512], [64, 1, 4096, 512], [], [], [], [], [], [], [], [], [], [], [], [], [],
         [], [], [], [], [], [64, 64, 1, 64], [64, 1, 4096, 64], [], [], [], [], []),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        ([64, 64, 1, 512], [64, 64, 1, 1]), (torch.bfloat16,),
        (64, 0.041666666666666664, 2147483647, 2147483647, "BNSD", 1, 0, 0, 0, 0, False, 0, 0, 0),
        ((), (), (), 64, 0.041666666666666664, 2147483647, 2147483647, "BNSD", 1, 0, 0, 0, 0, False, 0, 0, 0),
        132385024, None, 64, 2, 2, 2, id="case-64-64-1-512"),
    pytest.param(
        "case-64-32-1-512", 36, ([64, 32, 1, 512], [64, 1, 4096, 512], [64, 1, 4096, 512], [], [], [], [], [], [], [], [], [], [], [], [], [],
         [], [], [], [], [], [64, 32, 1, 64], [64, 1, 4096, 64], [], [], [], [], []),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        ([64, 32, 1, 512], [64, 32, 1, 1]), (torch.bfloat16,),
        (32, 0.041666666666666664, 2147483647, 2147483647, "BNSD", 1, 0, 0, 0, 0, False, 0, 0, 0),
        (None, None, None, 32, 0.041666666666666664, 2147483647, 2147483647, "BNSD", 1, 0, 0, 0, 0, False, 0, 0, 0),
        132385024, None, 32, 2, 2, 2, id="case-64-32-1-512"),
    pytest.param(
        "case-64-16-1-512", 36, ([64, 16, 1, 512], [64, 1, 4096, 512], [64, 1, 4096, 512], [], [], [], [], [], [], [], [], [], [], [], [], [],
         [], [], [], [], [], [64, 16, 1, 64], [64, 1, 4096, 64], [], [], [], [], []),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        ([64, 16, 1, 512], [64, 16, 1, 1]), (torch.bfloat16,),
        (16, 0.041666666666666664, 2147483647, 2147483647, "BNSD", 1, 0, 0, 0, 0, False, 0, 0, 0),
        (None, None, None, 16, 0.041666666666666664, 2147483647, 2147483647, "BNSD", 1, 0, 0, 0, 0, False, 0, 0, 0),
        132385024, None, 16, 2, 2, 4, id="case-64-16-1-512"),
    pytest.param(
        "case-64-128-2-512", 36, ([64, 128, 2, 512], [64, 1, 4096, 512], [64, 1, 4096, 512], [], [], [], [], [], [], [], [], [], [], [], [], [],
         [], [], [], [], [], [64, 128, 2, 64], [64, 1, 4096, 64], [], [], [], [], []),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        ([64, 128, 2, 512], [64, 128, 1, 1]), (torch.bfloat16,),
        (128, 0.041666666666666664, 2147483647, 2147483647, "BNSD", 1, 0, 0, 0, 0, False, 0, 0, 0),
        (None, None, None, 128, 0.041666666666666664, 2147483647, 2147483647, "BNSD", 1, 0, 0, 0, 0, False, 0, 0, 0),
        132385024, None, 32, 2, 2, 2, id="case-64-128-2-512"),
    pytest.param(
        "case-64-128-1-512", 36, ([64, 128, 1, 512], [64, 1, 4096, 512], [64, 1, 4096, 512], [], [], [], [], [], [], [], [], [], [], [], [], [],
         [], [], [], [], [], [64, 128, 1, 64], [64, 1, 4096, 64], [], [], [], [], []),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        ([64, 128, 1, 512], [64, 128, 1, 1]), (torch.bfloat16,),
        (128, 0.041666666666666664, 2147483647, 2147483647, "BNSD", 1, 0, 0, 0, 0, False, 0, 0, 0),
        (None, None, None, 128, 0.041666666666666664, 2147483647, 2147483647, "BNSD", 1, 0, 0, 0, 0, False, 0, 0, 0),
        132385024, None, 64, 2, 2, 2, id="case-64-128-1-512"),
    pytest.param(
        "case-64-1-16-512", 36, ([64, 1, 16, 512], [64, 4096, 1, 512], [64, 4096, 1, 512], [], [], [], [], [], [], [], [], [], [], [], [], [],
         [], [], [], [], [], [64, 1, 16, 64], [64, 4096, 1, 64], [], [], [], [], []),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        ([64, 1, 8192], [64, 16, 1, 1]), (torch.bfloat16,),
        (16, 0.041666666666666664, 2147483647, 2147483647, "BSND", 1, 0, 0, 0, 0, False, 0, 0, 0),
        (None, None, None, 16, 0.041666666666666664, 2147483647, 2147483647, "BSND", 1, 0, 0, 0, 0, False, 0, 0, 0),
        132385025, None, 16, 2, 2, 4, id="case-64-1-16-512"),
    pytest.param(
        "case-64-1-128-512", 36, ([64, 1, 128, 512], [64, 4096, 1, 512], [64, 4096, 1, 512], [], [], [], [], [], [], [], [], [], [], [], [], [],
         [], [], [], [], [], [64, 1, 128, 64], [64, 4096, 1, 64], [], [], [], [], []),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        ([64, 1, 65536], [64, 128, 1, 1]), (torch.bfloat16,),
        (128, 0.041666666666666664, 2147483647, 2147483647, "BSND", 1, 0, 0, 0, 0, False, 0, 0, 0),
        ((), (), (), 128, 0.041666666666666664, 2147483647, 2147483647, "BSND", 1, 0, 0, 0, 0, False, 0, 0, 0),
        132385025, None, 64, 2, 2, 2, id="case-64-1-128-512"),
    pytest.param(
        "case-256-64-1-512", 36, ([256, 64, 1, 512], [256, 1, 3584, 512], [256, 1, 3584, 512], [], [], [], [], [], [], [], [], [], [], [], [],
         [], [], [], [], [], [], [256, 64, 1, 64], [256, 1, 3584, 64], [], [], [], [], []),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        ([256, 64, 1, 512], [64, 16, 1, 1]), (torch.bfloat16,),
        (64, 0.041666666666666664, 2147483647, 2147483647, "BNSD", 1, 0, 0, 0, 0, False, 0, 0, 0),
        ((), (), (), 64, 0.041666666666666664, 2147483647, 2147483647, "BNSD", 1, 0, 0, 0, 0, False, 0, 0, 0),
        132385024, None, 64, 2, 2, 2, id="case-256-64-1-512"),
    pytest.param(
        "case-192-64-1-512", 36, ([192, 64, 1, 512], [192, 1, 3072, 512], [192, 1, 3072, 512], [], [], [], [], [], [], [], [], [], [], [], [],
         [], [], [], [], [], [], [192, 64, 1, 64], [192, 1, 3072, 64], [], [], [], [], []),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        ([192, 64, 1, 512], [192, 16, 1, 1]), (torch.bfloat16,),
        (64, 0.041666666666666664, 2147483647, 2147483647, "BNSD", 1, 0, 0, 0, 0, False, 0, 0, 0),
        ((), (), (), 64, 0.041666666666666664, 2147483647, 2147483647, "BNSD", 1, 0, 0, 0, 0, False, 0, 0, 0),
        132385024, None, 64, 2, 2, 2, id="case-192-64-1-512"),
    pytest.param(
        "case-192-1-64-512-4096", 36, ([192, 1, 64, 512], [192, 4096, 1, 512], [192, 4096, 1, 512], [], [], [], [], [], [], [], [], [], [], [], [],
         [], [], [], [], [], [], [192, 1, 64, 64], [192, 4096, 1, 64], [], [], [], [], []),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        ([192, 1, 32768], [192, 64, 1, 1]), (torch.bfloat16,),
        (64, 0.041666666666666664, 2147483647, 2147483647, "BSND", 1, 0, 0, 0, 0, False, 0, 0, 0),
        ((), (), (), 64, 0.041666666666666664, 2147483647, 2147483647, "BSND", 1, 0, 0, 0, 0, False, 0, 0, 0),
        132385025, None, 64, 2, 2, 2, id="case-192-1-64-512-4096"),
    pytest.param(
        "case-192-1-64-512-3584", 36, ([192, 1, 64, 512], [192, 3584, 1, 512], [192, 3584, 1, 512], [], [], [], [], [], [], [], [], [], [], [], [],
         [], [], [], [], [], [], [192, 1, 64, 64], [192, 3584, 1, 64], [], [], [], [], []),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        ([192, 1, 32768], [192, 64, 1, 1]), (torch.bfloat16,),
        (64, 0.041666666666666664, 2147483647, 2147483647, "BSND", 1, 0, 0, 0, 0, False, 0, 0, 0),
        ((), (), (), 64, 0.041666666666666664, 2147483647, 2147483647, "BSND", 1, 0, 0, 0, 0, False, 0, 0, 0),
        132385025, None, 64, 2, 2, 2, id="case-192-1-64-512-3584"),
    pytest.param(
        "case-192-1-64-512-3072", 36, ([192, 1, 64, 512], [192, 3072, 1, 512], [192, 3072, 1, 512], [], [], [], [], [], [], [], [], [], [], [], [],
         [], [], [], [], [], [], [192, 1, 64, 64], [192, 3072, 1, 64], [], [], [], [], []),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        ([192, 1, 32768], [192, 64, 1, 1]), (torch.bfloat16,),
        (64, 0.041666666666666664, 2147483647, 2147483647, "BSND", 1, 0, 0, 0, 0, False, 0, 0, 0),
        ((), (), (), 64, 0.041666666666666664, 2147483647, 2147483647, "BSND", 1, 0, 0, 0, 0, False, 0, 0, 0),
        132385025, None, 64, 2, 2, 2, id="case-192-1-64-512-3072"),
    # PYASC_TESTS_END
]
# yapf: enable


@pytest.mark.parametrize(
    "name, block_num, input_shapes, input_dtypes, output_shapes, output_dtypes, compile_params,"
    " runtime_params, tiling_key, tiling_params, g_group, s2_unroll, f_unroll, f_unroll_norm", test_cases)
def test_flash_attention_score(name, profiler, runs, block_num, input_shapes, input_dtypes, output_shapes,
                               output_dtypes, compile_params, runtime_params, tiling_key, tiling_params, g_group,
                               s2_unroll, f_unroll, f_unroll_norm, compile_only):
    q_shape = input_shapes[0]
    v_shape = input_shapes[1]
    if compile_params[4] == "BNSD":
        B, N, S1, D = q_shape
        S2 = v_shape[2]
        N_kv = v_shape[1]
        assert (v_shape[0] == B)
        assert (v_shape[3] == D)
    elif compile_params[4] == "BSND":
        B, S1, N, D = q_shape
        S2 = v_shape[1]
        N_kv = v_shape[2]
        assert (v_shape[0] == B)
        assert (v_shape[3] == D)
    else:
        raise RuntimeError("Wrong layout: " + compile_params[4])
    s1_step = S1
    # Bigger S2 tile (CANN s2BaseSize=128). L0B (64KB) is double-buffered by the unrolled
    # d_mm loop, so the B2 fragment area is bounded: 2 * d_mm * s2_step * 2B <= 64KB, i.e.
    # d_mm * s2_step <= 16384 elements. With s2_step=128 this gives d_mm=128 (CANN
    # MatmulK K-fragment); the [step2, d_v] l0v fragments reuse the l0k buffer via
    # reinterpret_cast.
    s2_step = 128
    d_mm_step = min(256, 16384 // s2_step)
    scale_factor = compile_params[1]
    d_v = 128
    assert D % d_v == 0
    torch_dtype = input_dtypes[0]
    assert torch_dtype == torch.float16 or torch_dtype == torch.bfloat16
    asctile_dtype = asctile.bfloat16 if torch_dtype == torch.bfloat16 else asctile.float16
    q = (torch.randn(B, N, S1, D, dtype=torch_dtype) * 0.5)
    k = torch.randn(B, N_kv, S2, D, dtype=torch_dtype)
    v = torch.randn(B, N_kv, S2, D, dtype=torch_dtype)
    o = torch.zeros(B, N, S1, D, dtype=torch_dtype)
    ws_o = torch.zeros(B * N * S1, D, dtype=torch.float32)
    with profiler.profile():
        for _ in range(runs):
            flash_attention_pyasc[block_num](q, k, v, o, ws_o, B, S1, S2, N, N_kv, D, s1_step, s2_step, d_mm_step,
                                             scale_factor, g_group, d_v, asctile_dtype, s2_unroll, f_unroll,
                                             f_unroll_norm)
    if compile_only:
        return
    o_ref = flash_attention_torch_reference(q, k, v, B, S1, S2, N, N_kv, D, s1_step, s2_step, scale_factor)
    o_ref = o_ref.to(torch_dtype)
    torch.testing.assert_close(o, o_ref, atol=1e-2, rtol=1e-2)
