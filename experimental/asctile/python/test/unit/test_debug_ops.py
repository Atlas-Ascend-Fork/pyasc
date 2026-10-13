# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys

from asc.experimental import asctile
from asc.runtime.jit import MockTensor
import pytest


class TestDeviceAssert:

    def test_assert(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            asctile.device_assert(asctile.block_idx() >= 0, "block_idx must be non-negative")

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_assert_without_msg(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            asctile.device_assert(asctile.block_idx() >= 0)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_assert_bool(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            asctile.device_assert(True, "always true")

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_invalid_msg_type(self, jit_test):

        @jit_test
        def kernel():
            asctile.device_assert(asctile.block_idx() >= 0, 123)

        with pytest.raises(TypeError, match="message"):
            kernel[1]()


class TestDevicePrint:

    def test_print_str(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            asctile.device_print("hello")

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_print_scalar(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            asctile.device_print(asctile.block_idx())

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_print_tensor(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([128], asctile.float32)
            asctile.device_print(x)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_print_global_tensor(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x = asctile.global_tensor(x_ptr, [128])
            asctile.device_print(x)

        kernel[1](MockTensor(asctile.float32))
        assert mock_launch.call_count == 1

    def test_print_mixed(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            x = zero_tile([128], asctile.float32)
            asctile.device_print("str", True, 1, 2.0, asctile.block_idx(), x, asctile.float32)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_print_custom_sep_end(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            asctile.device_print("a", "b", sep=", ", end=";\n")

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_print_sep_end_none(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            asctile.device_print("a", sep=None, end=None)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_print_no_args(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            asctile.device_print()

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_invalid_sep_type(self, jit_test):

        @jit_test
        def kernel():
            asctile.device_print("a", sep=123)

        with pytest.raises(TypeError, match="sep"):
            kernel[1]()

    def test_invalid_end_type(self, jit_test):

        @jit_test
        def kernel():
            asctile.device_print("a", end=123)

        with pytest.raises(TypeError, match="end"):
            kernel[1]()


class TestStaticAssert:

    def test_static_assert_true(self):
        asctile.static_assert(True, "ok")

    def test_static_assert_false(self):
        with pytest.raises(AssertionError, match="boom"):
            asctile.static_assert(False, "boom")

    def test_static_assert_false_without_msg(self):
        with pytest.raises(AssertionError):
            asctile.static_assert(False)

    def test_invalid_msg_type(self):
        with pytest.raises(TypeError, match="message"):
            asctile.static_assert(True, 123)


class TestStaticPrint:

    def test_static_print(self, capsys):
        asctile.static_print("hello", "world")
        assert "hello world" in capsys.readouterr().out

    def test_static_print_with_kwargs(self, capsys):
        asctile.static_print("a", "b", sep=", ", end=";\n", flush=True)
        assert "a, b;\n" in capsys.readouterr().out


class TestPrint:

    def test_print_string_on_device(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            print("hello from device")

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_print_runtime_value(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            print(asctile.block_idx())

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_print_flush_raises(self, jit_test):

        @jit_test
        def kernel():
            print("hello", flush=True)

        with pytest.raises(TypeError, match="flush"):
            kernel[1]()

    def test_print_file_raises(self, jit_test):

        @jit_test
        def kernel():
            print("hello", file=sys.stdout)

        with pytest.raises(TypeError, match="file"):
            kernel[1]()

    def test_print_file_flush_raises(self, jit_test):

        @jit_test
        def kernel():
            print("hello", file=sys.stdout, flush=True)

        with pytest.raises(TypeError, match="file.*flush"):
            kernel[1]()

    def test_print_no_args_on_device(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            print()

        kernel[1]()
        assert mock_launch.call_count == 1


class TestAssertStatement:

    def test_assert_bool(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            assert True, "always true"

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_assert_runtime(self, jit_test, mock_launch):

        @jit_test
        def kernel():
            assert asctile.block_idx() >= 0, "runtime check"

        kernel[1]()
        assert mock_launch.call_count == 1
