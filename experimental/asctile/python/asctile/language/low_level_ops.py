# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Iterable, Optional, Tuple, Type, TypeVar, overload

from asc._C.libpyasc import asctile, ir
from asc.language.core.dtype import DataType
from asc.language.core.ir_value import IRValue, materialize_ir_value as _mat
from asc.language.core.tensor import GlobalTensor as AscGlobalTensor, LocalTensor as AscLocalTensor
from asc.language.core.utils import global_builder, require_jit

from .global_tensor import GlobalTensor
from .local_tensor import LocalTensor
from .tensor_location import TensorLocation, TensorLocLike
from .utils import cast_tensor_location as cast_loc
from .validation import check_type, verify_location, verify_shape

T = TypeVar("T", bound=IRValue)


@require_jit
def empty(shape: Iterable[int], dtype: DataType, location: TensorLocLike = TensorLocation.Auto) -> LocalTensor:
    """
    Allocate an uninitialized local tensor.

    This is a low-level operation intended for interoperability with ``asc`` APIs or for temporary buffers whose
    contents are fully written before they are read. Unlike :py:func:`zeros`, it does not initialize tensor elements.

    Args:
        shape: The static shape of the tensor to allocate.
        dtype: The data type of the tensor.
        location: The memory location for the tensor. Default is ``TensorLocation.Auto``.

    Returns:
        LocalTensor: A new uninitialized tensor with the requested shape, data type, and memory location.

    Raises:
        RuntimeError: If the shape is empty or contains a non-positive dimension.
        TypeError: If the shape is not an iterable of integers, dtype is not a ``DataType``, or location is not
            a ``TensorLocation``-like value.

    Examples:
        Allocate an uninitialized 2D float tensor in UB memory: ::

            tile = asctile.empty([16, 32], asctile.float32, asctile.TensorLocation.UB)

        When a mask is known to select ``x``, use an uninitialized tensor as the unused ``else`` operand. This avoids
        the cost of zero-initializing values that are never consumed: ::

            other = asctile.empty([16, 32], x.dtype)
            selected = asctile.where(mask, x, other)
    """
    check_type("dtype", dtype, DataType)
    location = verify_location(location)
    shape = verify_shape(shape)
    builder = global_builder.get_ir_builder()
    ir_type = asctile.ir.get_asctile_LocalTensorType(shape, dtype.to_ir(), location)
    return cast_loc(LocalTensor(builder.create_tensor_EmptyOp(ir_type)))


@require_jit
def empty_like(tensor: LocalTensor, location: Optional[TensorLocLike] = None) -> LocalTensor:
    """
    Allocate an uninitialized local tensor with the same shape and data type as another tensor.

    The new tensor uses separate storage from ``tensor``. Its elements are not initialized and must be written before
    they are read.

    Args:
        tensor: The tensor whose shape and data type should be copied.
        location: The memory location for the new tensor. If None, uses ``tensor.location``.

    Returns:
        LocalTensor: A new uninitialized tensor with the same shape and data type as ``tensor``.

    Examples:
        Allocate an uninitialized output buffer with the same layout as an input tensor: ::

            out = asctile.empty_like(x, asctile.TensorLocation.UB)
    """
    location = tensor.location if location is None else verify_location(location)
    return empty(tensor.shape, tensor.dtype, location)


@require_jit
def inline(code: str, args: Optional[tuple] = None, before_function: bool = False) -> None:
    """
    Inject raw C++ (Ascend C) code into the generated kernel.

    The provided code string is emitted verbatim into the output Ascend C source at the current insertion point.
    Optional arguments can be passed to be substituted into the emitted code as IR values.

    Args:
        code: The raw C++ code string to inject into the generated kernel source. Use ``$0``, ``$1``, etc.
            as placeholders to reference arguments from the ``args`` list by index.
        args: Optional list of values to pass as arguments to the emitted code. Each value is materialized
            as an IR value and forwarded to the underlying verbatim operation. In the code string, use
            ``$<index>`` (e.g., ``$0``, ``$1``) to reference these arguments by their position in the list.
        before_function: If ``True``, emit the code before the current function body instead of at the current
            insertion point. Default is ``False``.

    Returns:
        None

    Examples:
        Inject constant declarations into the kernel: ::

            asctile.inline('''
                constexpr int32_t TOTAL_ROWS = 668;
                constexpr int32_t ELEMENTS_PER_ROW = 32;
            ''')

        Pass kernel arguments to inline code using ``$<index>`` placeholders: ::

            @asctile.jit
            def kernel(x_ptr: asctile.GlobalAddress, y_ptr: asctile.GlobalAddress, size: int):
                asctile.inline('''
                    auto input_ptr = $0;
                    auto output_ptr = $1;
                    int64_t length = $2;
                    AscendC::GlobalTensor<float> x_gm;
                    x_gm.SetGlobalBuffer(input_ptr);
                ''', [x_ptr, y_ptr, size])
    """
    args = None if args is None else [_mat(arg).to_ir() for arg in args]
    insert_point = None
    builder = global_builder.get_ir_builder()
    if before_function:
        current_function = builder.get_current_function()
        if current_function is not None:
            insert_point = builder.save_insertion_point()
            builder.set_insertion_point(current_function)
    builder.create_asctile_InlineOp(code, args)
    if insert_point is not None:
        builder.restore_insertion_point(insert_point)


@require_jit
def inline_vf(code: str, shape: Tuple[int, ...], dtype: DataType,
              inputs: Optional[Iterable[LocalTensor]] = None) -> LocalTensor:
    """
    Embed Ascend C VF (vector function) code within a kernel.

    This is an escape hatch for advanced users who need to express vector-fusion operations (e.g., Ascend C Reg calls)
    that are not covered by the built-in API. The provided code string is injected verbatim as the body of a
    ``__VEC_SCOPE__`` block in the generated Ascend C source.

    Tensors are referenced by positional placeholders: ``$0`` is always the output tensor, and ``$1``, ``$2``, ... refer
    to the input tensors in the order they appear in ``inputs``. Zero or more input tensors are allowed. Each
    placeholder will be replaced with a ``LocalTensor`` allocated for a corresponding tensor.

    All input tensors must reside in UB memory. The output tensor is always allocated in UB.

    Args:
        code: The raw Ascend C code string to embed (treated as a ``__VEC_SCOPE__`` body).
            Use ``$0`` for the output tensor and ``$1``, ``$2``, ... for input tensors.
        shape: The shape of the output tensor.
        dtype: The data type of the output tensor.
        inputs: An optional iterable of zero or more input tensors referenced as ``$1``, ``$2``, ... in the code.

    Returns:
        LocalTensor: A new UB tensor (``$0``) containing the result produced by the inline vector function.

    Raises:
        TypeError: If code is not a str, dtype is not a DataType, or any input is not a LocalTensor.
        RuntimeError: If any input tensor is not located in UB memory or shape is invalid.

    Examples:
        Embed an inline vector multiply-add (``x * y + z``) using Ascend C register API: ::

            out = asctile.inline_vf(
                '''
                auto* out_ptr = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());
                auto* x_ptr = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());
                auto* y_ptr = reinterpret_cast<__ubuf__ float*>($2.GetPhyAddr());
                auto* z_ptr = reinterpret_cast<__ubuf__ float*>($3.GetPhyAddr());
                AscendC::Reg::RegTensor<float> result_reg;
                . . .
                AscendC::Reg::MaskReg mask_reg = AscendC::Reg::UpdateMask<float>(mask);
                AscendC::Reg::DataCopy(x_reg, x_ptr);
                AscendC::Reg::DataCopy(y_reg, y_ptr);
                AscendC::Reg::Mul(xy_reg, x_reg, y_reg, mask_reg);
                . . .
                ''',
                x.shape, x.dtype, [x, y, z])

        In the example above, ``$0`` placeholder refers to a ``LocalTensor`` corresponding to ``out`` tensor;
        ``$1``, ``$2``, and ``$3`` refers to ``x``, ``y``, and ``z`` respectively.
    """
    check_type("code", code, str)
    check_type("dtype", dtype, DataType)
    shape = verify_shape(shape)
    ir_tiles = []
    if inputs is not None:
        for index, tensor in enumerate(inputs):
            check_type(f"inputs[{index}]", tensor, LocalTensor)
            ir_tiles.append(cast_loc(tensor, TensorLocation.UB).to_ir())
    ir_type = asctile.ir.get_asctile_LocalTensorType(shape, dtype.to_ir(), TensorLocation.UB)
    handle = global_builder.get_ir_builder().create_asctile_InlineVFOp(ir_type, ir_tiles, code)
    return cast_loc(LocalTensor(handle))


def create_unrealized_cast(ir_type: ir.Type, ir_cls: Type[T], operand: IRValue) -> T:
    handle = global_builder.get_ir_builder().create_UnrealizedConversionCastOp(ir_type, operand.to_ir())
    return ir_cls.from_ir(handle)


@overload
def to_asc(asctile_object: LocalTensor) -> AscLocalTensor:
    ...


@overload
def to_asc(asctile_object: GlobalTensor) -> AscGlobalTensor:
    ...


@overload
def to_asc(asctile_object: T) -> T:
    ...


@require_jit
def to_asc(asctile_object: IRValue) -> IRValue:
    """
    Convert an AscTile tensor to its low-level ``asc`` counterpart.

    The conversion inserts an unrealized cast and preserves the tensor's shape and data type. It does not copy tensor
    data or change its physical layout. Values that are not AscTile tensors are returned unchanged.

    Args:
        asctile_object: An AscTile ``LocalTensor`` or ``GlobalTensor``, or another IR value.

    Returns:
        IRValue: The corresponding low-level ``asc`` tensor, or the input value unchanged if it is not an AscTile
            tensor.

    Examples:
        Pass AscTile tensors to a low-level ``asc`` operation: ::

            dst = asctile.to_asc(asctile.empty_like(x, asctile.TensorLocation.UB))
            asc.mul(dst, asctile.to_asc(x), asctile.to_asc(y), x.size)

        Convert a global tensor for a low-level ``asc`` operation: ::

            x_gm = asctile.global_tensor(x_ptr, [16])
            x_asc = asctile.to_asc(x_gm)
            asc.data_cache_preload(x_asc, 0)
    """
    if isinstance(asctile_object, LocalTensor):
        ir_type = ir.get_local_tensor_type(asctile_object.dtype.to_ir(), asctile_object.shape)
        return create_unrealized_cast(ir_type, AscLocalTensor, asctile_object)
    if isinstance(asctile_object, GlobalTensor):
        ir_type = ir.get_global_tensor_type(asctile_object.dtype.to_ir(), asctile_object.shape)
        return create_unrealized_cast(ir_type, AscGlobalTensor, asctile_object)
    return asctile_object


@overload
def from_asc(asc_object: AscLocalTensor, *, location: TensorLocLike = TensorLocation.Auto) -> LocalTensor:
    ...


@overload
def from_asc(asc_object: AscGlobalTensor) -> GlobalTensor:
    ...


@overload
def from_asc(asc_object: T) -> T:
    ...


@require_jit
def from_asc(asc_object: IRValue, **kwargs) -> IRValue:
    """
    Convert a low-level ``asc`` tensor back to its AscTile counterpart.

    The conversion inserts an unrealized cast and preserves the tensor's shape and data type. It does not copy tensor
    data or change its physical layout. Values that are not low-level ``asc`` tensors are returned unchanged.

    Args:
        asc_object: A low-level ``asc`` ``LocalTensor`` or ``GlobalTensor``, or another IR value.
        location: An optional memory location for the resulting AscTile local tensor. Only applies to local tensors.
            Default is ``TensorLocation.Auto``.

    Returns:
        IRValue: The corresponding AscTile tensor, or the input value unchanged if it is not a low-level ``asc``
            tensor.

    Raises:
        TypeError: If location is supplied but is not a ``TensorLocation``-like value.

    Examples:
        Convert a low-level local tensor produced by :py:func:`to_asc`: ::

            dst = asctile.to_asc(asctile.empty_like(x, asctile.TensorLocation.UB))
            asc.mul(dst, asctile.to_asc(x), asctile.to_asc(y), x.size)
            return asctile.from_asc(dst, location=asctile.TensorLocation.UB)

        Convert a low-level global tensor produced by :py:func:`to_asc`: ::

            x_gm = asctile.global_tensor(x_ptr, [16])
            x_asc = asctile.to_asc(x_gm)
            x = asctile.from_asc(x_asc)
    """
    if isinstance(asc_object, AscLocalTensor):
        location = verify_location(kwargs.get("location", TensorLocation.Auto))
        ir_type = asctile.ir.get_asctile_LocalTensorType(asc_object.shape, asc_object.dtype.to_ir(), location)
        return create_unrealized_cast(ir_type, LocalTensor, asc_object)
    if isinstance(asc_object, AscGlobalTensor):
        ir_type = asctile.ir.get_asctile_GlobalTensorType(asc_object.shape, asc_object.dtype.to_ir())
        return create_unrealized_cast(ir_type, GlobalTensor, asc_object)
    return asc_object
