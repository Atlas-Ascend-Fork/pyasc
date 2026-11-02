# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import builtins
import math
from numbers import Number
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import numpy as np

from asc.language.core.dtype import DataType

from ..language.local_tensor import RoundMode
from ..language.validation import check_type
from .patch import override
from .types import GlobalAddress, GlobalTensor, LocalTensor, cast_data, dtype_to_numpy


class BlockContext:

    def __init__(self) -> None:
        self.block_idx: int = 0
        self.block_num: int = 1

    def set(self, block_idx: int, block_num: int) -> None:
        self.block_idx = int(block_idx)
        self.block_num = int(block_num)


block_context = BlockContext()


@override
def block_idx() -> int:
    return block_context.block_idx


@override
def block_num() -> int:
    return block_context.block_num


@override
def range(*args: Any, **kwargs: Any) -> builtins.range:
    del kwargs
    return builtins.range(*args)


@override
def static_range(*args: Any) -> builtins.range:
    return builtins.range(*args)


@override
def device_assert(test: Any, message: Optional[str] = None) -> None:
    if not test:
        raise AssertionError(message)


@override
def device_print(*values: Any, sep: Optional[str] = None, end: Optional[str] = None) -> None:
    print(*values, sep=sep, end=end)


@override
def global_tensor(base: GlobalAddress, shape: Iterable[Any], l2_cache: bool = True) -> GlobalTensor:
    del l2_cache
    return GlobalTensor(base, shape)


def clip(offsets: Iterable[Any], shape: Tuple[int, ...],
         bounds: Tuple[int, ...]) -> Tuple[Tuple[slice, ...], Tuple[slice, ...]]:
    offsets = tuple(map(int, offsets))
    if len(offsets) != len(bounds) or len(shape) != len(bounds):
        raise ValueError(f"offsets {offsets} and shape {shape} must have the rank of {bounds}")
    if any(off < 0 for off in offsets):
        raise IndexError(f"offsets {offsets} are out of bounds for shape {bounds}")
    counts = tuple(max(0, min(dim, total - off)) for off, dim, total in zip(offsets, shape, bounds))
    region = tuple(slice(off, off + cnt) for off, cnt in zip(offsets, counts))
    head = tuple(slice(None, cnt) for cnt in counts)
    return region, head


def verify_real_shape(real_shape: Iterable[int], shape: Tuple[int, ...]) -> Tuple[int, ...]:
    real_shape = tuple(map(int, real_shape))
    if len(real_shape) != len(shape) or any(real > dim for real, dim in zip(real_shape, shape)):
        raise ValueError(f"real_shape {real_shape} must match the rank of {shape} and not exceed it")
    return real_shape


@override
def copy_in(src: GlobalTensor, offsets: Iterable[Any], shape: Iterable[int], location: Any = None, *,
            real_shape: Optional[Iterable[int]] = None, pad_value: Optional[Union[int, float]] = None) -> LocalTensor:
    del location
    check_type("src", src, GlobalTensor)
    shape = tuple(map(int, shape))
    if pad_value is not None and real_shape is None:
        real_shape = shape
    if real_shape is not None and pad_value is None:
        pad_value = 0
    load_shape = shape if real_shape is None else verify_real_shape(real_shape, shape)
    region, head = clip(offsets, load_shape, src.shape)
    if real_shape is None:
        data = np.empty(shape, dtype=src.data.dtype)
    else:
        data = np.full(shape, pad_value, dtype=src.data.dtype)
    data[head] = src.data[region]
    return LocalTensor(data)


@override
def copy_out(src: LocalTensor, dst: GlobalTensor, offsets: Iterable[Any], *,
             real_shape: Optional[Iterable[int]] = None) -> None:
    check_type("src", src, LocalTensor)
    check_type("dst", dst, GlobalTensor)
    store_shape = src.shape if real_shape is None else verify_real_shape(real_shape, src.shape)
    region, head = clip(offsets, store_shape, dst.shape)
    dst.data[region] = src.data[head]


@override
def copy(src: LocalTensor, offsets: Iterable[Any], shape: Iterable[int], location: Any = None) -> LocalTensor:
    del location
    check_type("src", src, LocalTensor)
    shape = tuple(map(int, shape))
    region, head = clip(offsets, shape, src.shape)
    data = np.empty(shape, dtype=src.data.dtype)
    data[head] = src.data[region]
    return LocalTensor(data)


@override
def gather(src: GlobalTensor, offsets: Iterable[Any], dim: int, index: LocalTensor, check_bounds: bool = True,
           num_indices: Optional[int] = None, pad_value: Optional[Union[int, float]] = None) -> LocalTensor:
    dim = int(dim)
    count = index.shape[0] if num_indices is None else min(int(num_indices), index.shape[0])
    offsets = tuple(map(int, offsets))
    positions = offsets[dim] + index.data[:count].astype(np.int64)
    tail = src.data[offsets[:dim]]
    tail_shape = src.shape[dim + 1:]
    items_align = 32 // src.data.dtype.itemsize
    aligned = -(-tail_shape[-1] // items_align) * items_align
    result = np.full((index.shape[0], ) + tail_shape[:-1] + (aligned, ), pad_value if pad_value is not None else 0,
                     dtype=src.data.dtype)
    if check_bounds:
        valid = (index.data[:count] >= 0) & (positions < src.shape[dim])
        rows = np.flatnonzero(valid)
        positions = positions[valid]
    else:
        rows = np.arange(count)
    result[rows, ..., :tail_shape[-1]] = tail[positions]
    return LocalTensor(result)


@override
def scatter(src: LocalTensor, dim: int, index: LocalTensor, dst: GlobalTensor, offsets: Iterable[Any],
            check_bounds: bool = True, num_indices: Optional[int] = None) -> None:
    dim = int(dim)
    count = index.shape[0] if num_indices is None else min(int(num_indices), index.shape[0], src.shape[0])
    offsets = tuple(map(int, offsets))
    positions = offsets[dim] + index.data[:count].astype(np.int64)
    rows = src.data[:count][..., :dst.shape[-1]]
    if check_bounds:
        valid = (index.data[:count] >= 0) & (positions < dst.shape[dim])
        positions, rows = positions[valid], rows[valid]
    dst.data[offsets[:dim]][positions] = rows


def atomic_op(op: Callable[[Any, Any], Any], src: LocalTensor, dst: GlobalTensor, offsets: Iterable[Any]) -> None:
    store_shape = (1, ) * (dst.rank - src.rank) + src.shape
    region, head = clip(offsets, store_shape, dst.shape)
    dst.data[region] = op(dst.data[region], np.broadcast_to(src.data, store_shape)[head])


@override
def atomic_add(src: LocalTensor, dst: GlobalTensor, offsets: Iterable[Any]) -> None:
    atomic_op(np.add, src, dst, offsets)


@override
def atomic_max(src: LocalTensor, dst: GlobalTensor, offsets: Iterable[Any]) -> None:
    atomic_op(np.maximum, src, dst, offsets)


@override
def atomic_min(src: LocalTensor, dst: GlobalTensor, offsets: Iterable[Any]) -> None:
    atomic_op(np.minimum, src, dst, offsets)


@override
def zeros_acc(shape: Iterable[int], dtype: DataType, *, bias: Optional[LocalTensor] = None) -> LocalTensor:
    data = np.zeros(tuple(map(int, shape)), dtype=dtype_to_numpy(dtype))
    if bias is not None:
        data += bias.data.astype(data.dtype)
    return LocalTensor(data)


@override
def matmul_acc(acc: LocalTensor, input: LocalTensor, other: LocalTensor) -> None:
    check_type("acc", acc, LocalTensor)
    check_type("input", input, LocalTensor)
    check_type("other", other, LocalTensor)
    acc.data += input.data.astype(acc.data.dtype) @ other.data.astype(acc.data.dtype)


@override
def cast(input: Union[LocalTensor, Number], dtype: DataType,
         round_mode: Any = RoundMode.Default) -> Union[LocalTensor, Number]:
    if isinstance(input, LocalTensor):
        return LocalTensor(cast_data(input.data, dtype, round_mode))
    return cast_data(np.asarray(input), dtype).item()


@override
def concat(*inputs: LocalTensor) -> LocalTensor:
    return LocalTensor(np.concatenate([inp.data for inp in inputs], axis=0))


@override
def empty(shape: Iterable[int], dtype: DataType, location: Any = None) -> LocalTensor:
    del location
    return LocalTensor(np.empty(tuple(map(int, shape)), dtype=dtype_to_numpy(dtype)))


@override
def empty_like(tensor: LocalTensor, location: Any = None) -> LocalTensor:
    del location
    return LocalTensor(np.empty_like(tensor.data))


@override
def full(shape: Iterable[int], value: Union[int, float, np.generic], dtype: Optional[DataType] = None) -> LocalTensor:
    if dtype is None:
        dtype = DataType("int32") if isinstance(value, int) else DataType("float32")
    return LocalTensor(np.full(tuple(map(int, shape)), value, dtype=dtype_to_numpy(dtype)))


@override
def full_like(input: LocalTensor, value: Union[int, float, np.generic]) -> LocalTensor:
    return full(input.shape, value, input.dtype)


@override
def zeros(shape: Iterable[int], dtype: DataType = DataType("int32")) -> LocalTensor:
    return full(shape, 0, dtype)


@override
def zeros_like(input: LocalTensor) -> LocalTensor:
    return full(input.shape, 0, input.dtype)


@override
def where(mask: LocalTensor, src0: Union[LocalTensor, Number], src1: Union[LocalTensor, Number]) -> LocalTensor:
    values = [value.data if isinstance(value, LocalTensor) else value for value in (src0, src1)]
    return LocalTensor(np.where(mask.data, *values))


def bind_tensor_method(obj: Any = None, *, binary: bool = False) -> Any:

    def bind(fn: Callable[..., Any], name: Optional[str]) -> Callable[..., Any]:
        method_name = name or fn.__name__
        setattr(LocalTensor, method_name, fn)
        if binary:

            def swapped(self: Any, other: Any) -> Any:
                return fn(other, self)

            setattr(LocalTensor, "__r" + method_name[2:], swapped)
        return fn

    if callable(obj):
        return bind(obj, None)
    return lambda fn: bind(fn, obj)


def binary_op(op: Callable[[Any, Any], Any], lhs: Any, rhs: Any) -> LocalTensor:
    lhs = lhs.data if isinstance(lhs, LocalTensor) else lhs
    rhs = rhs.data if isinstance(rhs, LocalTensor) else rhs
    return LocalTensor(op(lhs, rhs))


@bind_tensor_method("__add__", binary=True)
@override
def add(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.add, input, other)


@bind_tensor_method("__and__", binary=True)
@override
def bitwise_and(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.bitwise_and, input, other)


@bind_tensor_method("__or__", binary=True)
@override
def bitwise_or(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.bitwise_or, input, other)


@bind_tensor_method("__xor__", binary=True)
@override
def bitwise_xor(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.bitwise_xor, input, other)


@bind_tensor_method("__truediv__", binary=True)
@override
def div(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.divide, input, other)


@bind_tensor_method("__lshift__")
@override
def left_shift(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.left_shift, input, other)


@bind_tensor_method("__matmul__", binary=True)
@override
def matmul(input: LocalTensor, other: LocalTensor, bias: Optional[LocalTensor] = None, *,
           hf32: bool = False) -> LocalTensor:
    del hf32
    result = input.data.astype(np.float32) @ other.data.astype(np.float32)
    if bias is not None:
        result = result + bias.data.astype(np.float32)
    return LocalTensor(result)


@override
def maximum(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.maximum, input, other)


@override
def minimum(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.minimum, input, other)


@bind_tensor_method("__mul__", binary=True)
@override
def mul(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.multiply, input, other)


@bind_tensor_method("__pow__", binary=True)
@override
def pow(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.power, input, other)


@bind_tensor_method("__rshift__")
@override
def right_shift(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.right_shift, input, other)


@bind_tensor_method("__sub__", binary=True)
@override
def sub(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.subtract, input, other)


@bind_tensor_method("__eq__")
@override
def equal(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.equal, input, other)


@bind_tensor_method("__gt__")
@override
def greater(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.greater, input, other)


@bind_tensor_method("__ge__")
@override
def greater_equal(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.greater_equal, input, other)


@bind_tensor_method("__lt__")
@override
def less(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.less, input, other)


@bind_tensor_method("__le__")
@override
def less_equal(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.less_equal, input, other)


@bind_tensor_method("__ne__")
@override
def not_equal(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.not_equal, input, other)


def unary_op(op: Callable[[Any], Any], input: Union[LocalTensor, Number]) -> LocalTensor:
    input = input.data if isinstance(input, LocalTensor) else input
    return LocalTensor(op(input))


@bind_tensor_method("__abs__")
@bind_tensor_method
@override
def abs(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(np.abs, input)


@bind_tensor_method("__invert__")
@override
def bitwise_not(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(np.invert, input)


@bind_tensor_method
@override
def ceil(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(np.ceil, input)


@bind_tensor_method
@override
def cos(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(np.cos, input)


@bind_tensor_method
@override
def cosh(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(np.cosh, input)


@bind_tensor_method
@override
def erf(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(np.vectorize(math.erf), input)


@bind_tensor_method
@override
def exp(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(np.exp, input)


@bind_tensor_method
@override
def exp2(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(np.exp2, input)


@bind_tensor_method
@override
def floor(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(np.floor, input)


@override
def layer_norm(input: LocalTensor, gamma: LocalTensor, beta: LocalTensor, epsilon: float,
               output_rstd: bool = True) -> Tuple[LocalTensor, LocalTensor, LocalTensor]:
    x = input.data.astype(np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    rstd = 1 / np.sqrt(var + epsilon)
    out = ((x - mean) * rstd * gamma.data.astype(np.float32) + beta.data.astype(np.float32))
    rows = input.shape[0] if input.rank == 2 else 1
    stat = rstd if output_rstd else var
    return (LocalTensor(out.astype(input.data.dtype)), LocalTensor(mean.reshape(rows)), LocalTensor(stat.reshape(rows)))


@bind_tensor_method
@override
def log(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(np.log, input)


@bind_tensor_method
@override
def log2(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(np.log2, input)


@bind_tensor_method("__neg__")
@override
def negative(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(np.negative, input)


@bind_tensor_method
@override
def relu(input: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.maximum, input, 0)


@override
def rms_norm(input: LocalTensor, gamma: LocalTensor, epsilon: float) -> LocalTensor:
    x = input.data.astype(np.float32)
    rms = np.sqrt(np.square(x).mean(axis=-1, keepdims=True) + epsilon)
    result = x / rms * gamma.data.astype(np.float32)
    return LocalTensor(result.astype(input.data.dtype))


@bind_tensor_method
@override
def rsqrt(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(lambda data: 1 / np.sqrt(data), input)


@bind_tensor_method
@override
def sin(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(np.sin, input)


@bind_tensor_method
@override
def sinh(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(np.sinh, input)


@override
def softmax(input: LocalTensor) -> LocalTensor:
    data = input.data.astype(np.float32)
    shifted = np.exp(data - data.max(axis=-1, keepdims=True))
    return LocalTensor((shifted / shifted.sum(axis=-1, keepdims=True)).astype(input.data.dtype))


@bind_tensor_method
@override
def sqrt(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(np.sqrt, input)


@bind_tensor_method
@override
def tan(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(np.tan, input)


@bind_tensor_method
@override
def tanh(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(np.tanh, input)


def reduce_op(op: Callable[..., Any], input: LocalTensor, dims: Tuple[int, ...], keep_dims: bool) -> Any:
    if not dims:
        value = op(input.data)
        return LocalTensor(np.full([1] * input.rank, value, dtype=input.data.dtype)) if keep_dims else value
    return LocalTensor(op(input.data, axis=dims, keepdims=keep_dims))


@bind_tensor_method("max")
@override
def reduce_max(input: LocalTensor, *dims: int, keep_dims: bool = False) -> Any:
    return reduce_op(np.max, input, dims, keep_dims)


@bind_tensor_method("min")
@override
def reduce_min(input: LocalTensor, *dims: int, keep_dims: bool = False) -> Any:
    return reduce_op(np.min, input, dims, keep_dims)


@bind_tensor_method("prod")
@override
def reduce_prod(input: LocalTensor, *dims: int, keep_dims: bool = False) -> Any:
    return reduce_op(np.prod, input, dims, keep_dims)


@bind_tensor_method("sum")
@override
def reduce_sum(input: LocalTensor, *dims: int, keep_dims: bool = False) -> Any:
    return reduce_op(np.sum, input, dims, keep_dims)


@override
def broadcast_tensors(*tensors: LocalTensor) -> Tuple[LocalTensor, ...]:
    shape = np.broadcast_shapes(*(tensor.shape for tensor in tensors))
    return tuple(LocalTensor(np.broadcast_to(tensor.data, shape)) for tensor in tensors)


@bind_tensor_method
@override
def broadcast_to(input: LocalTensor, *shape: Any) -> LocalTensor:
    return LocalTensor(np.broadcast_to(input.data, normalize_shape_args(shape)))


@bind_tensor_method
@override
def expand_dims(input: LocalTensor, *axis: Any) -> LocalTensor:
    return LocalTensor(np.expand_dims(input.data, tuple(map(int, axis))))


def normalize_shape_args(args: Tuple[Any, ...]) -> Tuple[int, ...]:
    shape = args[0] if len(args) == 1 and isinstance(args[0], Iterable) else args
    return tuple(map(int, shape))


@bind_tensor_method
@override
def ravel(input: LocalTensor) -> LocalTensor:
    return LocalTensor(input.data.reshape(-1))


@bind_tensor_method
@override
def reshape(input: LocalTensor, *shape: Any) -> LocalTensor:
    return LocalTensor(input.data.reshape(normalize_shape_args(shape)))


@bind_tensor_method
@override
def squeeze(input: LocalTensor, *axis: Any) -> LocalTensor:
    if axis:
        return LocalTensor(np.squeeze(input.data, tuple(map(int, axis))))
    return LocalTensor(np.squeeze(input.data))


@bind_tensor_method
@override
def transpose(input: LocalTensor, *axis: Any) -> LocalTensor:
    if not axis:
        axis = list(range(input.rank))
        axis[-1], axis[-2] = axis[-2], axis[-1]
    return LocalTensor(np.transpose(input.data, tuple(map(int, axis))))


@override
def reindex(src: LocalTensor, index: LocalTensor) -> LocalTensor:
    return LocalTensor(src.data.ravel()[index.data.astype(np.int64).ravel()].reshape(index.shape))


@override
def inline(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("'inline' is not supported by the interpreter")


@override
def inline_vf(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("'inline_vf' is not supported by the interpreter")


@override
def mask(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("'mask' is not supported by the interpreter")


@override
def from_asc(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("'from_asc' is not supported by the interpreter")


@override
def to_asc(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("'to_asc' is not supported by the interpreter")
