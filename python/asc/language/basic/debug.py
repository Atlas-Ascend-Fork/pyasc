# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any, List, Optional, overload

from ..core.dtype import DataType, KnownTypes
from ..core.ir_value import IRValue, PlainValue, materialize_ir_value
from ..core.tensor import GlobalTensor, LocalTensor
from ..core.utils import allow_jit, check_type, global_builder, require_jit
from .dump_tensor import dump_tensor, printf
from .sys_var import trap


def escape_percent(s: str) -> str:
    return s.replace("%", "%%")


@require_jit
def device_print(*values: Any, sep: Optional[str] = None, end: Optional[str] = None) -> None:
    """
    Print values during kernel execution on the device.

    Note:
        Device output is enabled when the kernel is compiled with ``debug=True``.
        Inside a JIT function, the built-in ``print`` call dispatches to this function.

    Example:
        ::

            @asc.jit(debug=True)
            def kernel():
                asc.device_print("block", asc.get_block_idx())
    """
    check_type("sep", sep, Optional[str])
    check_type("end", end, Optional[str])

    sep = escape_percent(sep) if sep is not None else " "
    end = escape_percent(end) if end is not None else "\n"
    format_parts: List[str] = []
    format_values: List[IRValue] = []
    format_specifiers = {
        DataType.Kind.Float: "%f",
        DataType.Kind.Int: "%d",
        DataType.Kind.UInt: "%u",
    }

    def flush() -> None:
        if format_parts:
            printf("".join(format_parts) + end, *format_values)
            format_parts.clear()
            format_values.clear()

    for value in values:
        if isinstance(value, LocalTensor):
            flush()
            dump_tensor(value, 0, value.get_size())
            continue
        if format_parts and sep:
            format_parts.append(sep)
        if isinstance(value, GlobalTensor):
            format_parts.append(f"GlobalTensor(addr=%p, dtype={value.dtype})")
            format_values.append(value.get_phy_addr())
        elif isinstance(value, PlainValue):
            format_specifier = format_specifiers.get(value.dtype.kind)
            if format_specifier is None:
                raise ValueError(f"There is no format specifier to print {value.dtype} value")
            format_parts.append(format_specifier)
            format_values.append(value)
        else:
            format_parts.append(escape_percent(str(value)))
    flush()


@overload
def static_print(*values: Any, sep: Optional[str] = None, end: Optional[str] = None, **kwargs) -> None:
    ...


@allow_jit
def static_print(*values: Any, **kwargs: Any) -> None:
    """
    Print values during JIT compilation on the host.

    Note:
        Host output is produced regardless of the ``debug`` option.
        Inside a JIT function, use this function explicitly; built-in ``print`` dispatches to :func:`device_print`.

    Example:
        ::

            @asc.jit
            def kernel(TILE: asc.ConstExpr[int]):
                asc.static_print("TILE =", TILE)
    """
    print(*values, **kwargs)


@require_jit
def device_assert(test: Any, message: Optional[str] = None) -> None:
    """
    Check a condition during kernel execution on the device.

    Note:
        Device assertions are intended for kernels compiled with ``debug=True``.
        Inside a JIT function, the built-in ``assert`` statement dispatches to this function.

    Example:
        ::

            @asc.jit(debug=True)
            def kernel(index: int):
                asc.device_assert(index >= 0, "index must be non-negative")
    """
    check_type("message", message, Optional[str])
    condition = materialize_ir_value(test, KnownTypes.int1) == 0
    builder = global_builder.get_ir_builder()
    insert_point = builder.save_insertion_point()
    op = builder.create_scf_IfOp(condition.to_ir(), [], with_else=False)
    then_block = op.get_then_block()
    builder.set_insertion_point_to_start(then_block)
    text = "Assertion failed" if message is None else f"Assertion failed: {message}"
    printf(escape_percent(text) + "\n")
    trap()
    builder.restore_insertion_point(insert_point)


@allow_jit
def static_assert(test: Any, message: Optional[str] = None) -> None:
    """
    Check a condition during JIT compilation on the host.

    Note:
        Host assertions run regardless of the ``debug`` option. The built-in ``assert`` statement inside a JIT function
        dispatches to :func:`device_assert`; use this function explicitly for compile-time checks.

    Example:
        ::

            @asc.jit
            def kernel(TILE: asc.ConstExpr[int]):
                asc.static_assert(TILE % 16 == 0, "TILE must be a multiple of 16")
    """
    if isinstance(test, IRValue):
        raise TypeError(f"static_assert expects a compile-time value, got {test!r}, use device_assert")
    check_type("message", message, Optional[str])
    if not test:
        raise AssertionError(message or "")
