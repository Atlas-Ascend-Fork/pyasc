# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from asc.experimental import asctile
from asc.runtime.jit import MockTensor
import pytest

from .helpers import all_locations


class TestInline:

    def test_inline(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            asctile.inline('constexpr int32_t x = 42;')

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_inline_with_args(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            asctile.inline('auto ptr = $0;', [x_ptr])

        kernel[1](MockTensor(asctile.float32))
        assert mock_launch.call_count == 1

    def test_inline_before_function(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            asctile.inline('constexpr int32_t x = 42;', before_function=True)

        kernel[1]()
        assert mock_launch.call_count == 1


class TestInlineVf:

    def test_without_inputs(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            asctile.inline_vf('// noop', [32, 64], asctile.float32)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_with_inputs(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([32, 64], asctile.float32)
            y = zero_tile([32, 64], asctile.float32)
            asctile.inline_vf('// noop', [32, 64], asctile.float32, [x, y])

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_invalid_code_type(self, jit_test):

        @jit_test
        def kernel():
            asctile.inline_vf(123, [32], asctile.float32)

        with pytest.raises(TypeError, match="code"):
            kernel[1]()

    def test_invalid_dtype_type(self, jit_test):

        @jit_test
        def kernel():
            asctile.inline_vf('// noop', [32], "invalid")

        with pytest.raises(TypeError, match="dtype"):
            kernel[1]()

    def test_invalid_input_type(self, jit_test):

        @jit_test
        def kernel():
            asctile.inline_vf('// noop', [32], asctile.float32, ["invalid"])

        with pytest.raises(TypeError, match="inputs"):
            kernel[1]()


@pytest.mark.parametrize("loc", [*all_locations, *(loc.name for loc in all_locations)])
def test_empty(jit_test, mock_launch, loc):

    @jit_test
    def kernel():
        asctile.empty([16, 32], asctile.float32, loc)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_empty_defaults_to_auto(jit_test, mock_launch):

    @jit_test
    def kernel():
        asctile.empty([16], asctile.float32)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_empty_like(jit_test, mock_launch, zero_tile):

    @jit_test
    def kernel():
        x = zero_tile([16, 32], asctile.float32, asctile.TensorLocation.UB)
        asctile.empty_like(x)
        asctile.empty_like(x, asctile.TensorLocation.L1)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_empty_invalid_dtype(jit_test):

    @jit_test
    def kernel():
        asctile.empty([16], "float32")

    with pytest.raises(TypeError, match="dtype"):
        kernel[1]()


@pytest.mark.parametrize("shape", ([], [0], ["16"]))
def test_empty_invalid_shape(jit_test, shape):

    @jit_test
    def kernel():
        asctile.empty(shape, asctile.float32)

    with pytest.raises((RuntimeError, TypeError, ValueError), match="shape"):
        kernel[1]()


def test_empty_invalid_location(jit_test):

    @jit_test
    def kernel():
        asctile.empty([16], asctile.float32, "invalid")

    with pytest.raises(ValueError, match="invalid"):
        kernel[1]()


def test_to_and_from_asc_local(jit_test, mock_launch):

    @jit_test
    def kernel():
        x = asctile.empty([16], asctile.float32)
        asc_x = asctile.to_asc(x)
        asctile.from_asc(asc_x)
        asctile.from_asc(asc_x, location=asctile.TensorLocation.L1)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_to_asc_global(jit_test, mock_launch):

    @jit_test
    def kernel(x_ptr: asctile.GlobalAddress):
        x = asctile.global_tensor(x_ptr, [16])
        asctile.to_asc(x)

    kernel[1](MockTensor(asctile.float32))
    assert mock_launch.call_count == 1


def test_to_and_from_asc_passthrough(jit_test, mock_launch):

    @jit_test
    def kernel():
        value = asctile.to_asc(0)
        asctile.from_asc(value)

    kernel[1]()
    assert mock_launch.call_count == 1
