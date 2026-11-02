# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

from math import prod
from types import MappingProxyType
from typing import Any, Iterable, SupportsInt, Tuple, Union

import numpy as np

from asc.language.core.dtype import DataType

from ..language.local_tensor import RoundMode
from ..language.tensor_location import TensorLocation
from ..language.validation import check_type
from .patch import override

dtype_to_numpy_map = MappingProxyType({
    "int1": np.dtype(np.bool_),
    "int8": np.dtype(np.int8),
    "int16": np.dtype(np.int16),
    "int32": np.dtype(np.int32),
    "int64": np.dtype(np.int64),
    "uint8": np.dtype(np.uint8),
    "uint16": np.dtype(np.uint16),
    "uint32": np.dtype(np.uint32),
    "uint64": np.dtype(np.uint64),
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
})

numpy_to_dtype_map = MappingProxyType({np_dtype: name for name, np_dtype in dtype_to_numpy_map.items()})


def dtype_to_numpy(dtype: Union[DataType, str]) -> np.dtype:
    check_type("dtype", dtype, (DataType, str))
    name = dtype if isinstance(dtype, str) else dtype.name
    if name in dtype_to_numpy_map:
        return dtype_to_numpy_map[name]
    raise ValueError(f"DataType '{name}' is not supported by the interpreter")


def numpy_to_dtype(dtype: np.dtype) -> DataType:
    check_type("dtype", dtype, np.dtype)
    name = numpy_to_dtype_map.get(np.dtype(dtype))
    if name is not None:
        return DataType(name)
    raise ValueError(f"dtype '{dtype}' is not supported by the interpreter")


round_mode_map = MappingProxyType({
    RoundMode.Rint: np.rint,
    RoundMode.Floor: np.floor,
    RoundMode.Ceil: np.ceil,
    RoundMode.Round: lambda data: np.trunc(data + np.copysign(0.5, data)),
    RoundMode.Trunc: np.trunc,
    RoundMode.NoRound: np.trunc,
    RoundMode.Default: np.trunc,
    RoundMode.Odd: lambda data: 2 * np.rint(data / 2 - 0.5) + 1,
})


def cast_data(data: Any, dtype: DataType, round_mode: Any = RoundMode.Default) -> Any:
    target = dtype_to_numpy(dtype)
    if data.dtype.kind == "f" and target.kind in "iub":
        data = round_mode_map[round_mode](data)
    if target.kind in "iu":
        info = np.iinfo(target)
        data = np.clip(data, info.min, info.max)
    return data.astype(target)


@override
class GlobalAddress:

    def __init__(self, storage: np.ndarray):
        check_type("storage", storage, np.ndarray)
        if not storage.flags["C_CONTIGUOUS"]:
            raise NotImplementedError("Only C-contiguous arrays are supported by the interpreter")
        self.storage = storage.ravel()
        self.dtype = numpy_to_dtype(self.storage.dtype)

    @property
    def size(self) -> int:
        return self.storage.size

    def __add__(self, offset: Any) -> GlobalAddress:
        offset = int(offset)
        if offset < 0:
            raise ValueError("Negative address offsets are not supported by the interpreter")
        return GlobalAddress(self.storage[offset:])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dtype={self.dtype}, size={self.size})"


@override
class GlobalTensor:

    def __init__(self, address: GlobalAddress, shape: Iterable[SupportsInt]):
        check_type("address", address, GlobalAddress)
        self.address = address
        self.shape = tuple(map(int, shape))
        if not self.shape:
            raise ValueError("Global tensor shape must not be empty")
        if any(dim <= 0 for dim in self.shape):
            raise ValueError(f"Global tensor shape must contain positive values, got {self.shape}")
        elems = prod(self.shape)
        if elems > address.size:
            raise ValueError(
                f"Global tensor of shape {self.shape} ({elems} elements) exceeds the storage of {address.size} "
                "elements")
        self.dtype = address.dtype
        self.data = address.storage[:elems].reshape(self.shape)

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return self.data.size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dtype={self.dtype}, shape={self.shape})"


@override
class LocalTensor:

    def __init__(self, data: np.ndarray):
        self.data = np.asarray(data)

    @property
    def dtype(self) -> DataType:
        return numpy_to_dtype(self.data.dtype)

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> LocalTensor:
        return self.transpose()

    def to(self, dtype_or_location: Union[DataType, TensorLocation, str],
           round_mode: Any = RoundMode.Default) -> LocalTensor:
        if isinstance(dtype_or_location, DataType):
            return LocalTensor(cast_data(self.data, dtype_or_location, round_mode))
        return self

    def __pos__(self) -> LocalTensor:
        return self

    def __floordiv__(self, other: Any) -> LocalTensor:
        return self / other

    def __rfloordiv__(self, other: Any) -> LocalTensor:
        return other / self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dtype={self.data.dtype}, shape={self.data.shape}, data={self.data})"
