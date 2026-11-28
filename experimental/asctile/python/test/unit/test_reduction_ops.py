# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass, field
from typing import Callable, Tuple

from asc.experimental import asctile
import pytest

from .helpers import all_dtypes


@dataclass(frozen=True)
class ReductionSpec:
    fn: Callable
    valid_dtypes: Tuple[asctile.DataType, ...]
    valid_scalar_dtypes: Tuple[asctile.DataType, ...] = field(default_factory=tuple)
    supports_scalar: bool = True

    def __str__(self):
        return self.fn.__name__


specs = (
    ReductionSpec(
        fn=asctile.reduce_sum,
        valid_dtypes=(asctile.int32, asctile.int64, asctile.float32),
        valid_scalar_dtypes=(asctile.int64, asctile.float16, asctile.float32),
    ),
    ReductionSpec(
        fn=asctile.reduce_max,
        valid_dtypes=(asctile.int8, asctile.int16, asctile.int32, asctile.int64, asctile.float16, asctile.bfloat16,
                      asctile.float32),
        valid_scalar_dtypes=(asctile.int16, asctile.int32, asctile.int64, asctile.float16, asctile.float32),
    ),
    ReductionSpec(
        fn=asctile.reduce_min,
        valid_dtypes=(asctile.int8, asctile.int16, asctile.int32, asctile.int64, asctile.float16, asctile.bfloat16,
                      asctile.float32),
        valid_scalar_dtypes=(asctile.int16, asctile.int32, asctile.int64, asctile.float16, asctile.float32),
    ),
    ReductionSpec(fn=asctile.reduce_prod, valid_dtypes=(asctile.float32, ), supports_scalar=False),
)


@pytest.mark.parametrize("spec, dtype", [(s, d) for s in specs for d in s.valid_dtypes])
@pytest.mark.parametrize("keep_dims", [False, True])
def test_reduce(jit_test, mock_launch, zero_tile, spec: ReductionSpec, dtype, keep_dims):

    @jit_test
    def kernel():
        x = zero_tile([32, 64], dtype)
        spec.fn(x, 0, keep_dims=keep_dims)

    kernel[1]()
    assert mock_launch.call_count == 1


@pytest.mark.parametrize("spec, dtype", [(s, d) for s in specs for d in s.valid_scalar_dtypes if s.supports_scalar])
def test_reduce_scalar(jit_test, mock_launch, zero_tile, spec: ReductionSpec, dtype):

    @jit_test
    def kernel():
        x = zero_tile([32, 64], dtype)
        spec.fn(x)

    kernel[1]()
    assert mock_launch.call_count == 1


@pytest.mark.parametrize("spec", specs)
def test_reduce_keep_dims_multidim(jit_test, mock_launch, zero_tile, spec: ReductionSpec):

    @jit_test
    def kernel():
        x = zero_tile([32, 64], spec.valid_dtypes[0])
        spec.fn(x, 0, 1, keep_dims=True)

    kernel[1]()
    assert mock_launch.call_count == 1


@pytest.mark.parametrize(
    "spec, dtype",
    [(s, d) for s in specs for d in all_dtypes if d not in s.valid_dtypes and d not in s.valid_scalar_dtypes])
def test_invalid_dtype(jit_test, zero_tile, spec: ReductionSpec, dtype):

    @jit_test
    def kernel():
        x = zero_tile([32, 64], dtype)
        spec.fn(x, 0)

    with pytest.raises(RuntimeError, match="dtype"):
        kernel[1]()


@pytest.mark.parametrize("spec", specs)
def test_repeating_dims(jit_test, zero_tile, spec: ReductionSpec):
    dtype = spec.valid_dtypes[0]

    @jit_test
    def kernel():
        x = zero_tile([32, 64], dtype)
        spec.fn(x, 0, 0)

    with pytest.raises(RuntimeError, match="Repeating"):
        kernel[1]()


@pytest.mark.parametrize("spec", specs)
def test_out_of_range_dim(jit_test, zero_tile, spec: ReductionSpec):
    dtype = spec.valid_dtypes[0]

    @jit_test
    def kernel():
        x = zero_tile([32, 64], dtype)
        spec.fn(x, 5)

    with pytest.raises(RuntimeError, match="between 0 and"):
        kernel[1]()


def test_reduce_prod_no_scalar(jit_test, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([32, 64], asctile.float32)
        asctile.reduce_prod(x)

    with pytest.raises(RuntimeError, match="not supported"):
        kernel[1]()


def test_non_int_dim(jit_test, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([32, 64], asctile.float32)
        asctile.reduce_sum(x, "invalid")

    with pytest.raises(TypeError, match="dimensions must be int"):
        kernel[1]()


def test_invalid_input_type(jit_test):

    @jit_test
    def kernel():
        asctile.reduce_sum("invalid", 0)

    with pytest.raises(TypeError, match="input"):
        kernel[1]()


def test_reduce_all_dims(jit_test, mock_launch, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([32, 64], asctile.float32)
        asctile.reduce_sum(x, 0, 1)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_reduce_all_unit_dims(jit_test, mock_launch, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([1, 1], asctile.float32)
        asctile.reduce_sum(x, 0, 1, keep_dims=True)

    kernel[1]()
    assert mock_launch.call_count == 1
