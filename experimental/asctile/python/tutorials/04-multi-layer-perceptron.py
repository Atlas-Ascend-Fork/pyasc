# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Multi-Layer Perceptron
======================

A fused Multi-Layer Perceptron (MLP) layer ``out = gelu(X @ W1) @ W2`` that mixes the two compute units of an Ascend
AI core: the **cube** unit runs the two matrix multiplications, while the two **vector** sub-cores of the same core
run the elementwise work in between and after them -- the gelu activation on the first matmul's result and the
float32 -> float16 cast and store of the second -- in parallel with the cube's next tile. The hidden activation
never leaves the chip.

.. currentmodule:: asc.experimental

In this tutorial you will learn about:

* The anatomy of an Ascend AI core: one cube sub-core plus two vector sub-cores, and the ``cv_ratio=2`` JIT option
  that puts both vector sub-cores to work.

* Sub-block SPMD: :py:func:`asctile.sub_block_idx` and :py:func:`asctile.sub_block_num`, and why the tile loop starts
  at ``block_idx() // sub_block_num()``.

* Feeding the cube straight from global memory: ``copy_in`` into L0A/L0B, with the required L1 hop inserted by the
  compiler.

* The ``L0C -> UB`` fixpipe copy with ``distrib=SplitByM``, which streams each matmul result out of the cube and
  splits its rows between the two vector sub-cores' Unified Buffers.

* Running the gelu activation (tanh approximation) and the quantisation cast (float32 -> float16) in UB -- on the
  vector unit -- so both units can work on different tiles at the same time.

* The ``UB -> L1`` copy with ``distrib=JoinByM``: the inverse of the split, which reassembles the activated halves
  into one L1 tile and hands it back to the cube as the left operand of the second matmul -- the intermediate never
  touches global memory.

* How the compiler splits this single kernel into cube (AIC) and vector (AIV) parts and inserts the cross-core
  synchronisation for you.
"""

# %%
# Motivation
# ----------
#
# An MLP is two matmuls with an elementwise activation wedged in between: ``out = gelu(X @ W1) @ W2``. The matmuls are
# cube work, but the gelu is not: the cube's L0C register has no exp or division unit, so the activation must run in
# UB on the vector hardware anyway -- the only question is whether it comes back to the cube for the second matmul
# or makes a full round-trip through global memory. If the MLP were written as separate ``@asctile.jit`` kernels,
# the hidden activation would travel ``GM -> matmul -> GM -> gelu -> GM -> matmul -> GM``. A plain kernel
# (``cv_ratio=1``) would leave the second vector sub-core of every core idle while all of it happens.
#
# An Ascend AI core is not one unit: on ``Ascend950PR_9599`` it is a cluster of *sub-cores* -- one **cube** sub-core
# (the matmul engine owning L1, L0A/L0B/L0C) and two **vector** sub-cores, each with its own Unified Buffer::
#
#     one AI core (Ascend950PR_9599)
#     +-------------------------------------------------------------+
#     | cube sub-core (AIC)   L1 -> L0A/L0B -> L0C   matmul         |
#     | vector sub-core 0 (AIV)   own UB   elementwise, cast, store |
#     | vector sub-core 1 (AIV)   own UB   elementwise, cast, store |
#     +-------------------------------------------------------------+
#
# ===== ====== ======================================================================
# Actor Role
# ===== ====== ======================================================================
# AIC   cube   sub-core: loads X/W1/W2 into L0A/L0B, accumulates both matmuls in L0C.
# AIV 0 vector sub-core 0: gelu, cast and store the rows ``[0, M/2)`` of each tile.
# AIV 1 vector sub-core 1: gelu, cast and store the rows ``[M/2, M)`` of each tile.
# ===== ====== ======================================================================
#
# With the ``cv_ratio=2`` JIT option the kernel becomes *cube-vector mixed* and the whole MLP fuses into one launch.
# The dataflow is a cube-vector ping-pong around the two matmuls: the first accumulator leaves L0C through the
# **fixpipe** unit, which copies it to UB and (for ``distrib=SplitByM``) delivers each vector sub-core the half of
# the rows it is responsible for; gelu runs on the halves; and a ``UB -> L1`` *join* -- the exact inverse of the
# split -- reassembles the tile in L1 for the cube's second matmul:
#
# ======= ===================== ==============================================
# Stage   What happens          Data path
# ======= ===================== ==============================================
# matmul1 cube: hidden = X @ W1 GM -> L0A/L0B (via L1) -> L0C
# gelu    AIV: activate halves  L0C --fixpipe(SplitByM)--> UB
# load    AIV: reassemble tile  UB --(JoinByM)--> L0A
# matmul2 cube: out = act @ W2  L1 -> L0A and GM -> L0B (via L1) -> L0C
# store   AIV: cast and store   L0C --fixpipe(SplitByM)--> UB -> GM
# ======= ===================== ==============================================
#
# The point of the exercise is that the hidden activation stays on-chip -- no GM round-trip between the two matmuls --
# while the two units hand off twice per tile and otherwise work in parallel: while the vector sub-cores gelu one
# tile's activation or drain a finished tile to GM, the cube multiplies ahead on the next tile (schematic; each unit
# waits only at the two hand-offs of the tile in flight)::
#
#     cube  : [t0 X@W1][t0 act@W2][t1 X@W1][t1 act@W2] ...
#     AIV 0 :          [t0 gelu]              [t0 store] ...
#     AIV 1 :          [t0 gelu]              [t0 store] ...
#
# You still write single kernel in Python. The compiler splits it into an AIC part (L1/L0 traffic and matmuls) and an
# AIV part (UB ops and stores) and inserts the cross-core synchronisation at the hand-offs -- the fixpipe copy and
# the join above are exactly such points.

# %%
# Compute Kernel
# --------------
#
# The kernel simplifies tutorial 03's matmul: instead of staging tiles in L1 by hand, operands load straight from GM
# into the cube's L0A/L0B registers and the compiler inserts the L1 hop -- tutorial 03's three nested K loops
# disappear. The *tile mapping* becomes a grid-stride loop that the sub-blocks of a core walk in lock-step, and the
# *epilogue* moves from the cube to the vector sub-cores -- twice. Between the matmuls a whole new vector stage
# appears: the gelu activation, followed by a join that returns the tile to the cube.
#
# Two options decorate the kernel:
#
# * ``cv_ratio=2`` builds a *dual-core* kernels out of this one kernel: the cube sub-core and both vector
#   sub-cores of every launched core execute it, each on the operations it owns. See
#   :py:attr:`asctile.CompileOptions.cv_ratio`.
#
#   - ``distrib=SplitByM`` or ``DistribMode.SplitByM`` makes the compiler split the tile's rows: sub-block 0 receives
#     rows [0, M/2) into its UB, sub-block 1 rows [M/2, M). Each sub-block then holds its own
#     [single_core_m // 2, h] half.
#   - A UB -> L1 copy with the ``DistribMode.JoinByM`` parameter, or of a value assigned inside the strategy,
#     reassembles the halves into the full [single_core_m, h] tile for the cube.
#
# * ``reuse_alloc=2`` (met in tutorial 02) reuses freed on-chip buffers across loop iterations, keeping the footprint
#   flat now that both matmul results are created per tile.
#
# See also: the ``distrib`` argument of :py:func:`asctile.copy` and :py:class:`asctile.DistribMode`.

from asc.experimental import asctile


# GELU, tanh approximation:
#   gelu(x) = x / (1 + exp(neg_sqrt_eight_over_pi * (x * tanh_approx_factor + x^3)))
@asctile.jit()
def gelu(x):
    tanh_approx_factor = 1.0 / 0.044715
    neg_sqrt_eight_over_pi = -1.595769121 * 0.044715
    cub = x * x * x
    tmp = x * tanh_approx_factor + cub
    tmp = tmp * neg_sqrt_eight_over_pi
    tmp = asctile.exp(tmp)
    tmp = tmp + 1
    return x / tmp


@asctile.jit(cv_ratio=2, reuse_alloc=2)
def mlp(x_ptr, w1_ptr, w2_ptr, out_ptr, x_shape: asctile.ConstExpr, w1_shape: asctile.ConstExpr,
        w2_shape: asctile.ConstExpr, single_core_m: asctile.ConstExpr[int],  # tile height one AI core computes
        single_core_n: asctile.ConstExpr[int],  # output-tile width one AI core computes
        quant_type: asctile.ConstExpr,  # dtype to store the result in (e.g. float16)
        manual_split: asctile.ConstExpr[bool],  # use manual or automatic code split for vectors
        ):

    m, k = x_shape
    _, h = w1_shape
    _, n = w2_shape
    x_gm = asctile.global_tensor(x_ptr, x_shape)
    w1_gm = asctile.global_tensor(w1_ptr, w1_shape)
    w2_gm = asctile.global_tensor(w2_ptr, w2_shape)
    out_gm = asctile.global_tensor(out_ptr, [m, n])

    # The output grid: ceildiv(m, single_core_m) x ceildiv(n, single_core_n) tiles. Unlike tutorial 03 (one core =
    # one tile), we walk it with a grid-stride loop, because an MLP usually has more tiles than cores.
    n_blocks = asctile.ceildiv(n, single_core_n)
    tiles_num = asctile.ceildiv(m, single_core_m) * n_blocks

    # In a dual-core kernel every *sub-block* runs it, and ``block_idx()`` counts sub-blocks on the vector cores:
    # block_idx = block * sub_block_num + sub_block_idx. Dividing by ``sub_block_num()`` -- 2 in vector code, 1 in
    # cube code -- maps the cube and both vector sub-cores of one core onto the same output tile. The two units must
    # agree on the tiling, and this is how they do it (``/`` on integers is integer division, as before).
    start = asctile.block_idx() / asctile.sub_block_num()
    for tile_idx in asctile.range(start, tiles_num, asctile.block_num()):
        m_off = single_core_m * (tile_idx // n_blocks)
        n_off = single_core_n * (tile_idx % n_blocks)

        # ---- first cube stage: hidden = X @ W1 ------------------------------------------------------------
        # Straight from GM into the cube's private operand registers: ``copy_in`` into L0A/L0B makes the compiler
        # insert the required L1 hop (tutorial 03 staged L1 by hand to make the flow visible). The hidden tile is
        # [single_core_m, h] -- the *whole* hidden width, because the second matmul needs every column of it.
        x_l0a = asctile.copy_in(x_gm, [m_off, 0], [single_core_m, k], asctile.TensorLocation.L0A)
        w1_l0b = asctile.copy_in(w1_gm, [0, 0], [k, h], asctile.TensorLocation.L0B)
        # ``@`` runs on the cube and yields a fresh float32 [single_core_m, h] L0C accumulator, one per tile.
        hidden = x_l0a @ w1_l0b

        # ---- GELU activation on the vector sub-cores
        if manual_split:
            # Explicitly mark the subtiles to be split by M on the L0C -> UB copy.
            hidden_ub = asctile.copy(hidden, location=asctile.TensorLocation.UB, distrib=asctile.DistribMode.SplitByM)

            # GELU, tanh approximation, applied to the sub-block's half in float32, as ordinary vector ops in UB
            act = gelu(hidden_ub)

            # Explicitly mark the subtiles to be joined on the UB -> L0A copy.
            act_l0a = asctile.copy(act.to(quant_type), location=asctile.TensorLocation.L0A,
                                   distrib=asctile.DistribMode.JoinByM)
        else:
            with asctile.cv_strategy(asctile.DistribMode.SplitByM):
                hidden_ub = asctile.copy(hidden, location=asctile.TensorLocation.UB)
                # GELU, tanh approximation, applied to the sub-block's half in float32, as ordinary vector ops in UB
                act = gelu(hidden_ub)
            # Values assigned inside the strategy stay split outside it; a UB -> L0A copy of such a value is
            # automatically marked JoinByM and reassembles the full [single_core_m, h] tile for the cube.
            act_l0a = asctile.copy(act.to(quant_type), location=asctile.TensorLocation.L0A)

        # ---- second cube stage: out = act @ W2 ------------------------------------------------------------
        # The activation returns from the join as one L0A tile, and the matching W2 slice loads straight from GM into
        # L0B. Another fresh float32 [single_core_m, single_core_n] L0C accumulator.
        w2_l0b = asctile.copy_in(w2_gm, [0, n_off], [h, single_core_n], asctile.TensorLocation.L0B)
        out = act_l0a @ w2_l0b

        # ---- epilogue: copy the results of the second matmul through UB using all sub-blocks
        if manual_split:
            result = asctile.copy(out, location=asctile.TensorLocation.UB,
                                  distrib=asctile.DistribMode.SplitByM).to(quant_type)

            asctile.copy_out(result, out_gm, [m_off + result.shape[0] * asctile.sub_block_idx(), n_off])
        else:
            with asctile.cv_strategy(asctile.DistribMode.SplitByM):
                result = asctile.copy(out, location=asctile.TensorLocation.UB).to(quant_type)
                asctile.copy_out(result, out_gm, [m_off, n_off])


# %%
# Launch and Verify
# -----------------
#
# We run ``Multi-Layer perceptron`` on a small float16:
#
# * Shapes: ``X = [M, K] = [128, 128]``, ``W1 = [K, H] = [128, 128]``, ``W2 = [H, N] = [128, 128]``, and
#   ``out = [M, N] = [128, 128]``. Inputs are float16, both matmuls accumulate in float32, the vector sub-cores run
#   the gelu in float32 and cast the activation and the result back to float16.
#
# * Output grid: ``single_core_m = single_core_n = 32`` gives ``ceildiv(128,32) x ceildiv(128,32) = 4 x 4 = 16``
#   tiles, each keeping its whole ``[32, 128]`` hidden activation on-chip. ``SplitByM`` halves each 32-row tile into
#   two 16-row halves (it requires M to be even).
#
# * ``block_num = 4`` launches four AI cores. With ``cv_ratio=2`` that is four cube sub-cores plus eight vector
#   sub-cores; each core walks ``16 / 4 = 4`` tiles of the grid.
#
# * Operands load whole into the cube's registers: the ``[32, 128]`` X tile and the ``[128, 128]`` W1 for the first
#   matmul, the ``[128, 32]`` W2 slice for the second -- the compiler inserts the L1 hop and the K/H reductions.

if __name__ == "__main__":
    import torch

    asctile.set_platform(asctile.Backend.Model, asctile.Platform.Ascend950PR_9599)
    torch.manual_seed(0)

    m, k, h, n = 128, 128, 128, 128
    single_core_m, single_core_n = 32, 32
    block_num = 4
    quant_type = asctile.float16

    dtype = torch.float16
    # X is zero-centred and scaled so the hidden pre-activations land in gelu's nonlinear region (with all-positive
    # inputs gelu would be ~identity) while the exp inside the tanh approximation stays finite. W1/W2 stay positive
    # so the outputs keep away from zero.
    x = (torch.rand((m, k), dtype=dtype) - 0.5) * 0.25
    w1 = torch.rand((k, h), dtype=dtype)
    w2 = torch.rand((h, n), dtype=dtype)

    # Reference: float16 torch matmuls with the same tanh-approximation gelu in between; only the summation order and
    # the intermediate rounding differ from the NPU.
    reference = torch.nn.functional.gelu(x @ w1, approximate="tanh") @ w2

    for manual_split in (False, True):  # automatic vector code split first, then the manual one
        out = torch.zeros((m, n), dtype=dtype)
        mlp[block_num](x, w1, w2, out, x.shape, w1.shape, w2.shape, single_core_m, single_core_n, quant_type,
                       manual_split)

        # Differences of a few ulp are expected: the NPU accumulates in its own order, rounds the activation to float16
        # its own way, and its exp unit rounds ties its own way.
        torch.testing.assert_close(out, reference, atol=1e-3, rtol=1e-3)
        max_diff = (out.to(torch.float32) - reference.to(torch.float32)).abs().max().item()
        split = "manual split" if manual_split else "automatic split"
        print(f"mlp({split}): PASSED (gelu({m}x{k} @ {k}x{h}) @ {h}x{n} -> {m}x{n}, blocks={block_num}, "
              f"tile={single_core_m}x{single_core_n}, max diff={max_diff:.2e})")
