# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any, List, Optional, overload

from asc.language.core.dtype import DataType, KnownTypes as KT
from asc.language.core.ir_value import IRValue, PlainValue, materialize_ir_value as _mat
from asc.language.core.utils import allow_jit, global_builder, require_jit

from .global_tensor import GlobalTensor
from .local_tensor import LocalTensor
from .validation import check_type


def escape_percent(s: str) -> str:
    return s.replace("%", "%%")


def get_format_specifier(dtype: DataType) -> str:
    if dtype.is_float():
        return "%f"
    elif dtype.is_signed():
        return "%d"
    elif dtype.is_unsigned():
        return "%u"
    raise ValueError(f"There is no format specifier to print {dtype} value")


@require_jit
def device_assert(test: Any, message: Optional[str] = None) -> None:
    """
    Check a condition during kernel execution on the device.

    If the condition is false, prints an error message with source location and stops execution.

    Args:
        test: Condition to check (boolean, comparison, or value convertible to bool).
        message: Optional error message to include in the report.

    Note:
        Only active when ``debug=True`` is set in the JIT decorator. The built-in ``assert``
        statement dispatches to this function inside JIT kernels.

    Examples:
        Assert that the block index is non-negative: ::

            @asctile.jit(debug=True)
            def kernel():
                asctile.device_assert(asctile.block_idx() >= 0, "block_idx must be non-negative")
    """
    check_type("message", message, Optional[str])
    message = escape_percent(message) if message is not None else ""
    global_builder.get_ir_builder().create_asctile_AssertOp(_mat(test, KT.int1).to_ir(), message)


@require_jit
def device_print(*values: Any, sep: Optional[str] = None, end: Optional[str] = None) -> None:
    """
    Print values during kernel execution on the device.

    Outputs values to the device console. Strings are printed as-is, scalars are formatted by type,
    and tensors are dumped in full.

    Args:
        values: Values to print (strings, scalars, or tensors).
        sep: Separator between values (default ``" "``).
        end: Line terminator (default ``"\n"``).

    Note:
        Only active when ``debug=True`` is set in the JIT decorator. The built-in ``print()``
        function dispatches to this function inside JIT kernels.

    Examples:
        Print a mix of strings, scalars, and tensors: ::

            @asctile.jit(debug=True)
            def kernel():
                x = asctile.zeros([128], asctile.float32)
                asctile.device_print("block", asctile.block_idx(), "tensor", x)
    """
    check_type("sep", sep, Optional[str])
    check_type("end", end, Optional[str])
    fmt_parts: List[str] = []
    ir_args: List[PlainValue] = []
    sep = escape_percent(sep) if sep is not None else " "
    end = escape_percent(end) if end is not None else "\n"
    builder = global_builder.get_ir_builder()

    def flush():
        nonlocal fmt_parts, ir_args
        if not fmt_parts:
            return
        fmt_string = "".join(fmt_parts) + end
        builder.create_asc_PrintfOp(fmt_string, [arg.to_ir() for arg in ir_args])
        fmt_parts = []
        ir_args = []

    for value in values:
        if isinstance(value, (GlobalTensor, LocalTensor)):
            flush()
            builder.create_asctile_DumpTensorOp(value.to_ir())
            continue
        if fmt_parts and sep:
            fmt_parts.append(sep)
        if isinstance(value, PlainValue):
            fmt_parts.append(get_format_specifier(value.dtype))
            ir_args.append(value)
        else:
            if isinstance(value, bool):
                value = int(value)
            fmt_parts.append(escape_percent(str(value)))
    flush()


@allow_jit
def static_assert(test: Any, message: Optional[str] = None) -> None:
    """
    Check a condition during JIT compilation on the host.

    If the condition is false, raises ``AssertionError`` and aborts compilation.

    Args:
        test: Condition to check (must be a compile-time value, not an IR value).
        message: Optional error message for the exception.

    Note:
        Always runs during compilation, regardless of the ``debug`` option. The built-in ``assert``
        statement does NOT dispatch to this function — it dispatches to :func:`device_assert` instead.
        Use this function explicitly for compile-time checks.

    Examples:
        Validate a tile size constant at compile time: ::

            @asctile.jit
            def kernel(TILE: asc.ConstExpr[int]):
                asctile.static_assert(TILE % 16 == 0, "TILE must be a multiple of 16")
    """
    if isinstance(test, IRValue):
        raise TypeError(f"static_assert expects a compile-time value, got {type(test).__name__}, use device_assert")
    check_type("message", message, Optional[str])
    if not test:
        raise AssertionError(message or "")


@overload
def static_print(*values: Any, sep: Optional[str] = None, end: Optional[str] = None, **kwargs) -> None:
    ...


@allow_jit
def static_print(*args: Any, **kwargs: Any) -> None:
    """
    Print values during JIT compilation on the host.

    Outputs values to the host console. Use this to trace compile-time constants and metadata.

    Args:
        *args: Values to print.
        **kwargs: Forwarded to the built-in :func:`print`.

    Note:
        Always runs during compilation, regardless of the ``debug`` option. The built-in ``print()``
        function does NOT dispatch to this function — it dispatches to :func:`device_print` instead.
        Use this function explicitly for compile-time output.

    Examples:
        Log the configured tile size at compile time: ::

            @asctile.jit
            def kernel(TILE: asc.ConstExpr[int]):
                asctile.static_print("compiling kernel with TILE =", TILE)
    """
    print(*args, **kwargs)
