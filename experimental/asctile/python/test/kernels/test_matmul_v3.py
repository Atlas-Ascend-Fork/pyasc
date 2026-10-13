# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from asc.experimental import asctile
import pytest
import torch

from ..target.helpers import parametrize_is_static
from ..target.matmul_v3 import FullLoad, Transpose, matmul_v3_kernel
"""
Each test case is a tuple of:
(core_num, tiling_data, dtype, a_transp, b_transp, full_load_mode, has_bias, double_buffering)

tiling_data = (m, n, k, m_L1, n_L1, k_L1, base_m, base_n, base_k):
- m, n, k       — logical matrix dimensions
- m_L1, n_L1, k_L1 — L1 tile sizes (one load from GM to L1 covers this many elements)
- base_m, base_n, base_k — L0 tile sizes (one copy from L1 to L0A/L0B)

double_buffering = (tile_uf, m_uf, n_uf, k_l1_uf, k_l0_uf):
unroll_factor for each loop level; value > 1 enables double-buffering at that level

Data transfer hierarchy tested:
- GM -> L1  : copy_in of [base_m, k_L1] / [k_L1, base_n] blocks (or full matrix in FullLoad)
- L1 -> L0A : copy of [base_m, base_k] from a_l1 (with optional .T for a_transpose)
- L1 -> L0B : copy of [base_k, base_n] from b_l1 (with optional .T for b_transpose)
- L0A x L0B : matmul_acc into accumulator in L0C
- L0C -> GM : copy_out of quantized accumulator to c_gm

FullLoad (preloads entire matrix into L1, skipping per-outer_k GM loads):
- NONE — standard tiled loading: both A and B are loaded per outer_k iteration
- A    — full A matrix loaded into L1 upfront; B still loaded per outer_k
- B    — full B matrix loaded into L1 upfront; A still loaded per outer_k

Group:
Tiles are traversed in a snake-order pattern grouped by m-blocks. A "group" is a
contiguous set of m-blocks processed together before advancing along the n-axis.
group_size = 4 (fixed in kernel). main_group = min(group_size, m_blocks).
main_row = number of complete group-rows; tail_group = remaining m-blocks in the
last partial row. Even rows traverse n left-to-right, odd rows right-to-left
(snake ordering), which improves L1 reuse between adjacent tiles.

Coverage axes:
- Transpose: none / a_transpose / b_transpose / both
- FullLoad: NONE / A / B, combined with transpose variants
- Tail dimensions: m, n, k not divisible by L1/base sizes
- Multi-core: core_num > 1 with tile distribution across cores
- Double-buffering: at tile, m, n, outer_k, inner_k levels
- Iteration counts: multiple outer_k, inner_k, m-iters, n-iters, tiles
- Data types: fp32 (with/without hf32), fp16, bf16
- Bias: with and without bias addition
- Minimal/edge sizes: m=1, n=1, k=1
"""


@pytest.mark.parametrize(
    "core_num, tiling_data, dtype, a_transp, b_transp, full_load_mode, has_bias, double_buffering", [
        # 1 tile, 1 outer_k, 1 inner_k; b_transpose, fp32, no bias
        (1, (16, 16, 64, 16, 16, 64, 16, 16, 64), torch.float32, Transpose.NONE, Transpose.L1, FullLoad.NONE, False,
         (1, 1, 1, 2, 2)),
        # 1 tile, 1 outer_k, 1 inner_k; tail k=13 (k_L1=16), n=1, fp32, no bias
        (1, (16, 1, 13, 16, 16, 16, 16, 16, 16), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # 4 tiles (2x2), 1 outer_k, 1 inner_k; fp32, bias
        (1, (32, 64, 64, 16, 32, 64, 16, 32, 64), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.NONE, True,
         (1, 1, 1, 1, 1)),
        # 2 tiles (2x1), 1 outer_k, 1 inner_k; tail m=31 (m_L1=16), b_transpose, fp16, no bias
        (1, (31, 128, 4, 16, 128, 16, 16, 128, 16), torch.float16, Transpose.NONE, Transpose.L1, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # 3 tiles (3x1), 1 outer_k, 1 inner_k; a_transpose, bf16, no bias
        (1, (48, 16, 16, 16, 16, 16, 16, 16, 16), torch.bfloat16, Transpose.L1, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # 6 tiles (2x3), 1 outer_k, 1 inner_k; tail n=33 (n_L1=16), 2 cores, b_transpose, fp16, no bias
        (2, (32, 33, 160, 16, 16, 160, 16, 16, 160), torch.float16, Transpose.NONE, Transpose.L1, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # 4 tiles (1x4), 1 outer_k, 1 inner_k; 2 cores, fp32, no bias; inner_k double-buffered
        (2, (32, 128, 32, 32, 32, 32, 32, 32, 32), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 2)),
        # 2 tiles (2x1), 1 outer_k, 1 inner_k; tail m=33, n=63, 2 cores, b_transpose, fp32, bias
        (2, (33, 63, 32, 32, 64, 32, 32, 64, 32), torch.float32, Transpose.NONE, Transpose.L1, FullLoad.NONE, True,
         (1, 1, 1, 1, 1)),
        # 2 tiles (2x1), 1 outer_k, 3 inner_k (k_L1=48, base_k=16); fp16, no bias; inner_k double-buffered
        (1, (32, 16, 48, 16, 16, 48, 16, 16, 16), torch.float16, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 2)),
        # 2 tiles (1x2), 1 outer_k, 1 inner_k; 2 cores, both transpose, bf16, no bias
        (2, (48, 64, 64, 48, 32, 64, 48, 32, 64), torch.bfloat16, Transpose.L1, Transpose.L1, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # 2 tiles (1x2), 2 outer_k (k=80, k_L1=64), 1 inner_k; tail n=17, fp32, no bias
        (1, (16, 17, 80, 16, 16, 64, 16, 16, 64), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # 2 tiles (1x2), 1 outer_k, 1 inner_k; 2 cores, fp32, no bias
        (2,
         (32, 64, 160, 32, 32, 160, 32, 32, 160), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # 2 tiles (1x2), 1 outer_k, 1 inner_k; tail n=77 (n_L1=64), 2 cores, fp16, no bias
        (2,
         (32, 77, 128, 32, 64, 128, 32, 64, 128), torch.float16, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # 1 tile, 1 outer_k, 3 inner_k (k_L1=48, base_k=16); n=1, fp32, bias
        (1, (16, 1, 48, 16, 16, 48, 16, 16, 16), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.NONE, True,
         (1, 1, 1, 1, 1)),
        # 1 tile, 1 outer_k, 1 inner_k; 2 cores, b_transpose, bf16, bias; inner_k double-buffered
        (2, (32, 96, 64, 32, 96, 64, 32, 96, 64), torch.bfloat16, Transpose.NONE, Transpose.L1, FullLoad.NONE, True,
         (1, 1, 1, 1, 2)),
        # 8 tiles (1x8), 1 outer_k, 3 inner_k (k_L1=48, base_k=16); fp32, no bias
        (1, (16, 256, 48, 16, 32, 48, 16, 32, 16), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # 1 tile, 2 outer_k (k=128, k_L1=64), 1 inner_k; both transpose, fp16, no bias
        (1, (64, 16, 128, 64, 16, 64, 64, 16, 64), torch.float16, Transpose.L1, Transpose.L1, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # 1 tile, 1 outer_k, 1 inner_k; 2 cores, b_transpose, fp16, no bias
        (2, (64, 80, 160, 64, 80, 160, 64, 80, 160), torch.float16, Transpose.NONE, Transpose.L1, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # 1 tile, 1 outer_k, 1 inner_k; 2 cores, fp32, no bias
        (2, (64, 64, 16, 64, 64, 16, 64, 64, 16), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # 3 tiles (1x3), 1 outer_k, 1 inner_k; tail n=77 (n_L1=32), fp32, no bias
        (1, (16, 77, 64, 16, 32, 64, 16, 32, 64), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # 3 outer_k (k=96, k_L1=32), 1 inner_k; fp16, no bias; outer_k and inner_k double-buffered
        (1, (16, 16, 96, 16, 16, 32, 16, 16, 16), torch.float16, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 2, 2)),
        # 1 outer_k, 5 inner_k (k_L1=80, base_k=16); bf16, no bias; inner_k double-buffered
        (1, (16, 16, 80, 16, 16, 80, 16, 16, 16), torch.bfloat16, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 2)),
        # 3 m iters (m_L1=48, base_m=16), 1 outer_k, 1 inner_k; fp32, no bias; m double-buffered
        (1, (48, 16, 16, 48, 16, 16, 16, 16, 16), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 2, 1, 1, 1)),
        # 3 n iters (n_L1=48, base_n=16), 1 outer_k, 1 inner_k; fp16, no bias; n double-buffered
        (1, (16, 48, 16, 16, 48, 16, 16, 16, 16), torch.float16, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 2, 1, 1)),
        # 3 tiles (3x1), 1 outer_k, 1 inner_k; fp32, no bias; tile double-buffered
        (1, (48, 16, 16, 16, 16, 16, 16, 16, 16), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (2, 1, 1, 1, 1)),
        # 9 tiles (3x3), 5 per core, 1 outer_k, 1 inner_k; 2 cores, bf16, no bias; tile double-buffered
        (2, (48, 48, 16, 16, 16, 16, 16, 16, 16), torch.bfloat16, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (2, 1, 1, 1, 1)),
        # 3 outer_k (k=144, k_L1=48), 3 inner_k (base_k=16); fp16, no bias; outer_k and inner_k double-buffered
        (1, (16, 16, 144, 16, 16, 48, 16, 16, 16), torch.float16, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 2, 2)),
        # 2 m + 2 n + 2 outer_k, double-buffered outer_k and inner_k; a_transpose, fp32, bias
        (1, (32, 32, 64, 32, 32, 32, 16, 16, 16), torch.float32, Transpose.L1, Transpose.NONE, FullLoad.NONE, True,
         (1, 1, 1, 2, 2)),
        # 3 outer_k (k=96, k_L1=32), FullLoad.A; fp32, bias; outer_k and inner_k double-buffered
        (1, (16, 16, 96, 16, 16, 32, 16, 16, 16), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.A, True,
         (1, 1, 1, 2, 2)),
        # 5 inner_k (k_L1=80, base_k=16), FullLoad.B; fp16, no bias; inner_k double-buffered
        (1, (16, 16, 80, 16, 16, 80, 16, 16, 16), torch.float16, Transpose.NONE, Transpose.NONE, FullLoad.B, False,
         (1, 1, 1, 1, 2)),
        # 3 outer_k (k=96, k_L1=32), b_transpose; bf16, no bias; outer_k and inner_k double-buffered
        (1, (16, 16, 96, 16, 16, 32, 16, 16, 16), torch.bfloat16, Transpose.NONE, Transpose.L1, FullLoad.NONE, False,
         (1, 1, 1, 2, 2)),
        # 1 tile, 1 outer_k, 1 inner_k; 2 cores, a_transpose, bf16, bias
        (2, (64, 48, 80, 64, 48, 80, 64, 48, 80), torch.bfloat16, Transpose.L1, Transpose.NONE, FullLoad.NONE, True,
         (1, 1, 1, 1, 1)),
        # 1 tile, 3 outer_k (k=100, k_L1=48), 3 inner_k (base_k=16); m=1, FullLoad.A, fp32, bias
        (1, (1, 16, 100, 16, 16, 48, 16, 16, 16), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.A, True,
         (1, 1, 1, 1, 1)),
        # 2 tiles (1x2), 1 outer_k, 1 inner_k; bf16, no bias
        (1, (32, 128, 64, 32, 64, 64, 32, 64, 64), torch.bfloat16, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # 1 tile, 1 outer_k, 1 inner_k; fp16, no bias
        (1, (96, 128, 128, 96, 128, 128, 96, 128, 128), torch.float16, Transpose.NONE, Transpose.NONE, FullLoad.NONE,
         False, (1, 1, 1, 1, 1)),
        # 1 tile, 1 outer_k, 3 inner_k (k_L1=96, base_k=32); FullLoad.B, fp32, no bias
        (1, (64, 128, 96, 160, 128, 96, 160, 128, 32), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.B, False,
         (1, 1, 1, 1, 1)),
        # 1 tile, 1 outer_k, 1 inner_k; b_transpose, FullLoad.B, fp16, no bias
        (1, (16, 16, 128, 16, 16, 128, 16, 16, 128), torch.float16, Transpose.NONE, Transpose.L1, FullLoad.B, False,
         (1, 1, 1, 1, 1)),
        # 1 tile, 1 outer_k, 1 inner_k; both transpose, FullLoad.B, fp16, no bias
        (1, (16, 16, 128, 16, 16, 128, 16, 16, 128), torch.float16, Transpose.L1, Transpose.L1, FullLoad.B, False,
         (1, 1, 1, 1, 1)),
        # 1 tile, 2 outer_k (k=128, k_L1=64), 4 inner_k (base_k=16); b_transpose, FullLoad.B, fp32, bias
        (1, (64, 128, 128, 64, 128, 64, 64, 128, 16), torch.float32, Transpose.NONE, Transpose.L1, FullLoad.B, True,
         (1, 1, 1, 1, 1)),
        # 1 tile, 1 outer_k, 1 inner_k; m=4, b_transpose, FullLoad.A, fp16, no bias
        (1, (4, 128, 64, 16, 128, 64, 16, 128, 64), torch.float16, Transpose.NONE, Transpose.L1, FullLoad.A, False,
         (1, 1, 1, 1, 1)),
        # 1 tile, 1 outer_k, 1 inner_k; fp32, bias
        (1, (64, 64, 64, 64, 64, 64, 64, 64, 64), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.NONE, True,
         (1, 1, 1, 1, 1)),
        # FullLoad.A + a_transpose: 1 tile, 1 outer_k, 1 inner_k; fp16, no bias
        (1, (16, 16, 64, 16, 16, 64, 16, 16, 64), torch.float16, Transpose.L1, Transpose.NONE, FullLoad.A, False,
         (1, 1, 1, 1, 1)),
        # FullLoad.B + b_transpose: 1 tile, 1 outer_k, 1 inner_k; bf16, no bias
        (1, (16, 16, 64, 16, 16, 64, 16, 16, 64), torch.bfloat16, Transpose.NONE, Transpose.L1, FullLoad.B, False,
         (1, 1, 1, 1, 1)),
        # FullLoad.A + b_transpose: 1 tile, 1 outer_k, 1 inner_k; fp32, bias
        (1, (16, 16, 64, 16, 16, 64, 16, 16, 64), torch.float32, Transpose.NONE, Transpose.L1, FullLoad.A, True,
         (1, 1, 1, 1, 1)),
        # FullLoad.B + a_transpose: 1 tile, 1 outer_k, 1 inner_k; fp16, bias
        (1, (16, 16, 64, 16, 16, 64, 16, 16, 64), torch.float16, Transpose.L1, Transpose.NONE, FullLoad.B, True,
         (1, 1, 1, 1, 1)),
        # Tail in all dimensions: m=31, n=33, k=47; 2 cores, fp16, no bias
        (2, (31, 33, 47, 16, 16, 16, 16, 16, 16), torch.float16, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # Minimal sizes: m=1, n=1, k=1; fp32, no bias
        (1, (1, 1, 1, 16, 16, 16, 16, 16, 16), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # Max unroll_factor: all double-buffered; 3 outer_k, 3 inner_k, fp16, no bias
        (1, (16, 16, 96, 16, 16, 32, 16, 16, 16), torch.float16, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (2, 2, 2, 2, 2)),
        # Both transpose + FullLoad.A + bias: 1 tile, 1 outer_k, 1 inner_k; fp32
        (1, (16, 16, 64, 16, 16, 64, 16, 16, 64), torch.float32, Transpose.L1, Transpose.L1, FullLoad.A, True,
         (1, 1, 1, 1, 1)),
        # Both transpose + FullLoad.B + bias: 1 tile, 1 outer_k, 1 inner_k; fp32
        (1, (16, 16, 64, 16, 16, 64, 16, 16, 64), torch.float32, Transpose.L1, Transpose.L1, FullLoad.B, True,
         (1, 1, 1, 1, 1)),
        # Grouping: m_blocks=4 (m=64, m_L1=16), main_group=4, main_row=0, tail_group=4; fp32, no bias
        (1, (64, 16, 16, 16, 16, 16, 16, 16, 16), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # Grouping: m_blocks=5 (m=80, m_L1=16), main_group=4, main_row=0, tail_group=5; fp16, no bias
        (1, (80, 16, 16, 16, 16, 16, 16, 16, 16), torch.float16, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # Grouping + snake: m_blocks=8 (m=128, m_L1=16), main_group=4, main_row=1, tail_group=4; bf16, bias
        (1, (128, 16, 16, 16, 16, 16, 16, 16, 16), torch.bfloat16, Transpose.NONE, Transpose.NONE, FullLoad.NONE, True,
         (1, 1, 1, 1, 1)),
        # Grouping + snake multi-row: m_blocks=12 (m=192, m_L1=16), main_group=4, main_row=2; fp32, no bias
        (1, (192, 16, 16, 16, 16, 16, 16, 16, 16), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # Grouping + snake + n_blocks>1: m_blocks=8, n_blocks=2 (m=128, n=32, m_L1=16, n_L1=16); fp16, no bias
        (1, (128, 32, 16, 16, 16, 16, 16, 16, 16), torch.float16, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # Grouping + snake + outer_k>1: m_blocks=8, 2 outer_k (m=128, k=32, k_L1=16); fp32, no bias
        (1, (128, 16, 32, 16, 16, 16, 16, 16, 16), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 2, 2)),
        # Grouping + a_transpose: m_blocks=5 (m=80, m_L1=16), a_transpose; bf16, no bias
        (1, (80, 16, 16, 16, 16, 16, 16, 16, 16), torch.bfloat16, Transpose.L1, Transpose.NONE, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # Grouping + b_transpose: m_blocks=4 (m=64, m_L1=16), b_transpose; fp16, no bias
        (1, (64, 16, 16, 16, 16, 16, 16, 16, 16), torch.float16, Transpose.NONE, Transpose.L1, FullLoad.NONE, False,
         (1, 1, 1, 1, 1)),
        # Grouping + FullLoad.A: m_blocks=4 (m=64, m_L1=16), FullLoad.A; fp32, bias
        (1, (64, 16, 16, 16, 16, 16, 16, 16, 16), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.A, True,
         (1, 1, 1, 1, 1)),
        # Grouping + FullLoad.B: m_blocks=5 (m=80, m_L1=16), FullLoad.B; fp16, no bias
        (1, (80, 16, 16, 16, 16, 16, 16, 16, 16), torch.float16, Transpose.NONE, Transpose.NONE, FullLoad.B, False,
         (1, 1, 1, 1, 1)),
    ])
@parametrize_is_static()
def test_matmul_v3(profiler, runs, is_static, core_num, tiling_data, dtype, a_transp, b_transp, full_load_mode,
                   has_bias, double_buffering):
    quant_type = asctile.float32
    if dtype == torch.float16:
        quant_type = asctile.float16
    elif dtype == torch.bfloat16:
        quant_type = asctile.bfloat16
    m, n, k, m_L1, n_L1, k_L1, base_m, base_n, base_k = tiling_data
    a_shape = (m, k) if a_transp == Transpose.NONE else (k, m)
    b_shape = (k, n) if b_transp == Transpose.NONE else (n, k)
    full_load_tile_m = asctile.ceildiv(m, base_m) * base_m
    full_load_tile_k = asctile.ceildiv(k, k_L1) * k_L1
    full_load_tile_n = asctile.ceildiv(n, base_n) * base_n
    enable_hf32_mode = dtype == torch.float32
    low, high = (-1, 1) if not enable_hf32_mode else (0, 1)
    a = (high - low) * torch.rand(a_shape, dtype=dtype) + low
    b = (high - low) * torch.rand(b_shape, dtype=dtype) + low
    c = torch.zeros((m, n), dtype=dtype)
    bias = (high - low) * torch.rand([n], dtype=dtype) + low
    with profiler.profile():
        for _ in range(runs):
            matmul_v3_kernel[core_num](a, b, c, bias, asctile.ConstExpr(m) if is_static else m,
                                       asctile.ConstExpr(n) if is_static else n,
                                       asctile.ConstExpr(k) if is_static else k, m_L1, n_L1, k_L1, base_m, base_n,
                                       base_k, a_transp, b_transp, full_load_mode, quant_type, enable_hf32_mode,
                                       has_bias, double_buffering, False, full_load_tile_m, full_load_tile_k,
                                       full_load_tile_n, distrib_mode=None)
    if a_transp != Transpose.NONE:
        a = a.T
    if b_transp != Transpose.NONE:
        b = b.T
    c_ref = a.to(torch.float32) @ b.to(torch.float32)
    if has_bias:
        c_ref = c_ref + bias
    c_ref = c_ref.to(dtype)
    torch.testing.assert_close(c, c_ref, atol=1e-03, rtol=1e-03)
