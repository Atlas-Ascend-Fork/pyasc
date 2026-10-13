# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import itertools
from typing import Dict

from asc._C import ir
from asc.language.basic.sys_var import get_block_idx, get_block_num
from asc.language.core.dtype import KnownTypes
from asc.language.core.ir_value import IRHandle, PlainValue
from asc.language.core.utils import global_builder, require_jit

from .context_manager import ContextManager
from .memory_ops import DistribMode
from .validation import check_type


@require_jit
def block_idx() -> PlainValue:
    """
    Returns the current block (NPU core) index.

    In the AscTile programming model, kernels are executed across multiple NPU blocks (cores). This function returns the
    index of the current block, which can be used to determine which portion of the data to process.

    Returns:
        PlainValue: The current block index (0-based)

    Examples:
        Get the current block index to compute the data offset: ::

            idx = asctile.block_idx()
            offset = idx * TILE_SIZE
            tile = asctile.copy_in(x_gm, [offset], [TILE_SIZE])
    """
    return get_block_idx()


@require_jit
def block_num() -> PlainValue:
    """
    Returns the total number of blocks (NPU cores) allocated for the kernel.

    This function returns the total number of NPU blocks that are executing the kernel, which was specified when
    launching the kernel.

    Returns:
        PlainValue: The total number of blocks

    Examples:
        Use block count to compute a stride across blocks: ::

            idx = asctile.block_idx()
            n_blocks = asctile.block_num()
            stride = n_blocks * TILE_SIZE
    """
    return get_block_num()


@require_jit
def sub_block_idx() -> PlainValue:
    """
    Returns the current sub-block index of vector or cube unit on the AI core.

    This function is useful to distinguish between vector sub-cores if the platform has multiple of them, especially
    when the kernel employs both vector and cube units. For example, there are 2 vector sub-cores and 1 cube sub-core
    on Ascend950PR_9599 chip.

    Returns:
        PlainValue: The current sub-block index

    Examples:
        Use with block_idx to compute starting iteration in a loop: ::

            idx = asctile.block_idx()
            num = asctile.sub_block_num()
            start = idx / num
            for i in asctile.range(start, total, step=asctile.block_num()):
                ...
    """
    return PlainValue(global_builder.get_ir_builder().create_asc_GetSubBlockIdxOp(KnownTypes.int_.to_ir()))


@require_jit
def sub_block_num() -> PlainValue:
    """
    Returns the number of vector or cube sub-cores on the AI core.

    This function returns the count of sub-cores available on the current AI core. For example, on Ascend950PR_9599
    chip there are 2 vector sub-cores and 1 cube sub-core, so this function returns 2 when called from vector code
    and 1 when called from cube code.

    Returns:
        PlainValue: The number of sub-cores on the AI core

    Examples:
        Use with block_idx to compute starting iteration in a loop: ::

            idx = asctile.cast(asctile.block_idx(), asctile.int32)
            num = asctile.cast(asctile.sub_block_num(), asctile.int32)
            start = idx / num
            for i in asctile.range(start, total, step=asctile.block_num()):
                ...
    """
    return PlainValue(global_builder.get_ir_builder().create_asc_GetSubBlockNumOp(KnownTypes.int_.to_ir()))


class CVStrategyContext(ContextManager):

    def __init__(self, split: DistribMode) -> None:
        super().__init__()
        check_type("split", split, DistribMode)
        if split not in (DistribMode.SplitByM, DistribMode.SplitByN):
            raise ValueError(f"Splitting by axis must be requested, got {split.value}")
        self.split = split

    def __enter__(self) -> None:
        builder = global_builder.get_ir_builder()
        self.old_insertion_point = global_builder.get_ir_builder().save_insertion_point()
        self.block = ir.Block()
        builder.set_insertion_point_to_start(self.block)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        builder = global_builder.get_ir_builder()
        builder.create_asctile_YieldOp(list(self.yields.values()))
        builder.restore_insertion_point(self.old_insertion_point)
        ir_types = list(value.get_type() for value in self.yields.values())
        self.op = builder.create_asctile_CVStrategyOp(ir_types, self.split)
        block = builder.create_block(self.op.get_region(0))
        self.block.merge_block_before(block)
        builder.restore_insertion_point(self.old_insertion_point)

    def handle_yieldables(self, defined, redefined) -> None:
        self.yields = {name: value for name, value in itertools.chain(defined.items(), redefined.items())}

    def map_to_restore(self) -> Dict[str, IRHandle]:
        return {name: self.op.get_result(i) for i, name in enumerate(self.yields)}


@require_jit
def cv_strategy(split: DistribMode) -> CVStrategyContext:
    """
    [Experimental] Declare a split strategy for vector sub-blocks.

    This context manager is intended for kernels launched with ``cv_ratio=2``. It marks a 2D copy from ``L0C`` to
    ``UB`` and its dependent operations for splitting: ``SplitByM`` halves the first (M) axis, while ``SplitByN``
    halves the second (N) axis. Internally, each vector sub-block would receive and process one half of the tensor.

    Supported **vector** scenarios:

    * elementwise operations (:py:func:`add`, :py:func:`exp`, and other),
    * reduction operations (e.g. :py:func:`reduce_sum`, :py:func:`reduce_max`, and other),
    * reshape operations (:py:func:`reshape`, :py:func:`expand_dims`, :py:func:`ravel`, :py:func:`squeeze`),
    * broadcast operations (:py:func:`broadcast_to`, :py:func:`broadcast_tensors`).

    Supported **data transfer** operations: :py:func:`copy_out` (UB-to-GM), :py:func:`copy` (L0C-to-UB, UB-to-L1).
    Other operations are not eligible for splitting.

    Args:
        split: The axis along which to split tensor work. Must be ``DistribMode.SplitByM`` or ``DistribMode.SplitByN``.

    Raises:
        TypeError: If ``split`` is not a ``DistribMode``
        ValueError: If ``split`` does not request splitting by an axis

    Note:
        For every :py:func:`copy` operation with L0C-to-UB context inside, the copied tensor must have rank 2.
        For ``SplitByM``, its M axis must be a multiple of 2. For ``SplitByN``, its N axis must be a multiple of 32.
        The resulting tensors of all dependent operations must satisfy these requirements as well.
        In addition, these operations are not allowed to change the dimension along which the tensor is split.

    Examples:
        Split a matmul result by M, reduce each sub-block's partial result, and broadcast it to the bigger shape: ::

            addend = asctile.copy_in(input_tensor, offsets=[0, 0], shape=[32, 64])
            with asctile.cv_strategy(asctile.DistribMode.SplitByM):  # actual shape | internally split shape
                matmul = asctile.copy(a @ b, location="UB")          # [32, 64]     | [16, 64]
                elwise = (matmul + addend) * 3                       # [32, 64]     | [16, 64]
                reduce = elwise.sum(1)                               # [32]         | [16]
                rshape = reduce.expand_dims(1)                       # [32, 1]      | [16, 1]
                brcast = rshape.broadcast_to(32, 128)                # [32, 128]    | [16, 128]
                asctile.copy_out(brcast, output_tensor, [0, 0])      # each sub-block copies [16, 128] half of [32, 128]

        Split a matmul result by N and use it both inside and outside of the ``cv_strategy`` context: ::

            with asctile.cv_strategy(asctile.DistribMode.SplitByN):
                c_l0 = asctile.matmul(a, b)
                c_ub = c_l0.to("UB") * 3.0
                asctile.copy_out(c_ub, output_tensor, [0, 0])
            c_l1 = asctile.copy(c_ub, location=asctile.TensorLocation.L1)  # similar to 'c_ub.to("L1")'
    """
    return CVStrategyContext(split)
