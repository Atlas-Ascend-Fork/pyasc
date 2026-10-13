# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from asc.experimental import asctile
from asc.runtime.jit import MockTensor, MockValue
import pytest


class TestGlobalTensor:

    def test_static_shape(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [64, 128])
            asctile.static_assert(x_gm.rank == 2)
            asctile.static_assert(x_gm.shape.is_static())
            asctile.static_assert(len(x_gm.shape) == 2)
            asctile.static_assert(list(x_gm.shape) == [64, 128])

        kernel[1](MockTensor(asctile.float32))
        assert mock_launch.call_count == 1

    def test_dynamic_shape(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress, n: int):
            x_gm = asctile.global_tensor(x_ptr, [n, 128])
            asctile.static_assert(not x_gm.shape.is_static())
            asctile.static_assert(x_gm.shape.is_dynamic_dim(0))
            asctile.static_assert(not x_gm.shape.is_dynamic_dim(1))

        kernel[1](MockTensor(asctile.float32), MockValue(asctile.int32))
        assert mock_launch.call_count == 1

    def test_shape_getitem(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [64, 128])
            x_gm.shape[0]
            x_gm.shape[1]
            x_gm.shape[-1]

        kernel[1](MockTensor(asctile.float32))
        assert mock_launch.call_count == 1

    def test_shape_getitem_dynamic(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress, n: int):
            x_gm = asctile.global_tensor(x_ptr, [n, 128])
            x_gm.shape[0]
            x_gm.shape[1]

        kernel[1](MockTensor(asctile.float32), MockValue(asctile.int32))
        assert mock_launch.call_count == 1

    def test_shape_index_out_of_range(self, jit_test):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [64, 128])
            x_gm.shape[5]

        with pytest.raises(IndexError, match="out of range"):
            kernel[1](MockTensor(asctile.float32))

    def test_shape_negative_index_out_of_range(self, jit_test):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [64, 128])
            x_gm.shape[-3]

        with pytest.raises(IndexError, match="out of range"):
            kernel[1](MockTensor(asctile.float32))

    def test_invalid_base_type(self, jit_test):

        @jit_test
        def kernel():
            asctile.global_tensor("invalid", [128])

        with pytest.raises(TypeError, match="base"):
            kernel[1]()

    def test_invalid_shape_values(self, jit_test):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            asctile.global_tensor(x_ptr, ["invalid"])

        with pytest.raises(TypeError, match="must be int"):
            kernel[1](MockTensor(asctile.float32))
