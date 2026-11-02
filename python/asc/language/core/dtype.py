# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..._C import ir
from .utils import global_builder


class DataType:

    class Kind(Enum):
        Any = ""
        Int = "int"
        Float = "float"
        UInt = "uint"

    @dataclass
    class BuiltinType:
        factory: Optional[str]
        value_factory: Optional[str]
        bitwidth: int
        kind: DataType.Kind

    support_types = {
        "void": BuiltinType("get_none_type", None, None, Kind.Any),
        "int1": BuiltinType("get_i1_type", "get_i1", 8, Kind.Int),
        "int8": BuiltinType("get_i8_type", "get_i8", 8, Kind.Int),
        "int16": BuiltinType("get_i16_type", "get_i16", 16, Kind.Int),
        "int32": BuiltinType("get_i32_type", "get_i32", 32, Kind.Int),
        "int64": BuiltinType("get_i64_type", "get_i64", 64, Kind.Int),
        "uint8": BuiltinType("get_ui8_type", "get_ui8", 8, Kind.UInt),
        "uint16": BuiltinType("get_ui16_type", "get_ui16", 16, Kind.UInt),
        "uint32": BuiltinType("get_ui32_type", "get_ui32", 32, Kind.UInt),
        "uint64": BuiltinType("get_ui64_type", "get_ui64", 64, Kind.UInt),
        "float16": BuiltinType("get_f16_type", "get_f16", 16, Kind.Float),
        "bfloat16": BuiltinType("get_bf16_type", "get_bf16", 16, Kind.Float),
        "float32": BuiltinType("get_f32_type", "get_f32", 32, Kind.Float),
        "float64": BuiltinType("get_f64_type", "get_f64", 64, Kind.Float),
        "float8_e4m3fn": BuiltinType("get_f8e4m3fn_type", "get_f8e4m3fn", 8, Kind.Float),
        "float8_e5m2": BuiltinType("get_f8e5m2_type", "get_f8e5m2", 8, Kind.Float),
    }

    def __init__(self, name: str):
        self.name = name
        try:
            self.kind = self.support_types[name].kind
            self.bitwidth = self.support_types[name].bitwidth
        except KeyError:
            raise ValueError(f"Unsupported type {name}") from None

    def __eq__(self, other: DataType) -> bool:
        if not isinstance(other, DataType):
            return False
        return self.name == other.name

    def __neq__(self, other: DataType) -> bool:
        return not (self == other)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_ir(cls, ir_type: ir.Type):
        name = ir_type.get_py_name()
        if name:
            return cls(name)
        raise ValueError(f"IR type doesn't have a Pythonic name: {ir_type}")

    def is_int(self) -> bool:
        return self.is_signed() or self.is_unsigned()

    def is_float(self) -> bool:
        return self.kind == self.Kind.Float

    def is_numeric(self) -> bool:
        return self.is_int() or self.is_float()

    def is_signed(self) -> bool:
        return self.kind == self.Kind.Int

    def is_unsigned(self) -> bool:
        return self.kind == self.Kind.UInt

    def is_void(self) -> bool:
        return self == void

    def sizeof(self) -> int:
        if self.bitwidth is None:
            raise ValueError(f"DataType does not have a bitwidth: {self.name}")
        div, mod = divmod(self.bitwidth, 8)
        if mod != 0:
            raise ValueError(f"DataType bitwidth does not fit an integer number of bytes: {self.bitwidth}")
        return div

    def to_ir(self) -> ir.Type:
        try:
            builder = global_builder.get_ir_builder()
            factory = getattr(builder, self.support_types[self.name].factory)
            return factory()
        except IndexError:
            raise ValueError(f"Unsupported DataType name: {self.name}") from None


# Predefined data types
void = DataType("void")
int1 = DataType("int1")
int8 = DataType("int8")
int16 = DataType("int16")
int32 = DataType("int32")
int64 = DataType("int64")
float16 = DataType("float16")
bfloat16 = DataType("bfloat16")
float32 = DataType("float32")
float64 = DataType("float64")
uint8 = DataType("uint8")
uint16 = DataType("uint16")
uint32 = DataType("uint32")
uint64 = DataType("uint64")
bit = int1
bool_ = int8
int_ = int32
half = float16
float_ = float32
double = float64
float8_e4m3fn = DataType("float8_e4m3fn")
float8_e5m2 = DataType("float8_e5m2")


class KnownTypes:
    void = void
    int1 = int1
    int8 = int8
    int16 = int16
    int32 = int32
    int64 = int64
    float16 = float16
    bfloat16 = bfloat16
    float32 = float32
    float64 = float64
    uint8 = uint8
    uint16 = uint16
    uint32 = uint32
    uint64 = uint64
    bit = bit
    bool_ = int8
    int_ = int32
    half = half
    float_ = float_
    double = double
    float8_e4m3fn = float8_e4m3fn
    float8_e5m2 = float8_e5m2
