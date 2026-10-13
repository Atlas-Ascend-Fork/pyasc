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

valid_dtypes = (asctile.int8, asctile.int16, asctile.int32, asctile.int64, asctile.float16, asctile.bfloat16,
                asctile.float32)
index_dtypes = (asctile.int8, asctile.int16, asctile.int32, asctile.int64)
copy_in_locs = (asctile.TensorLocation.UB, asctile.TensorLocation.L1, asctile.TensorLocation.L0A,
                asctile.TensorLocation.L0B)


class TestCopyIn:

    @pytest.mark.parametrize("dtype", valid_dtypes)
    @pytest.mark.parametrize("loc", copy_in_locs)
    def test_copy_in(self, jit_test, mock_launch, dtype, loc):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [64, 128])
            asctile.copy_in(x_gm, [0, 0], [64, 128], loc)

        kernel[1](MockTensor(dtype))
        assert mock_launch.call_count == 1

    def test_scalar(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [128])
            asctile.copy_in(x_gm, [0])

        kernel[1](MockTensor(asctile.float32))
        assert mock_launch.call_count == 1

    def test_with_padding(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [256])
            asctile.copy_in(x_gm, [0], [128], real_shape=[64], pad_value=0.0)

        kernel[1](MockTensor(asctile.float32))
        assert mock_launch.call_count == 1

    def test_pad_value_without_real_shape(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [128])
            asctile.copy_in(x_gm, [0], [128], pad_value=0.0)

        kernel[1](MockTensor(asctile.float32))
        assert mock_launch.call_count == 1

    def test_real_shape_without_pad_value(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [128])
            asctile.copy_in(x_gm, [0], [128], real_shape=[64])

        kernel[1](MockTensor(asctile.float32))
        assert mock_launch.call_count == 1

    def test_real_shape_exceeds_tensor(self, jit_test):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [64])
            asctile.copy_in(x_gm, [0], [128], real_shape=[256])

        with pytest.raises(RuntimeError, match="exceeds"):
            kernel[1](MockTensor(asctile.float32))

    def test_invalid_src_type(self, jit_test):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            asctile.copy_in("invalid", [0], [128])

        with pytest.raises(TypeError, match="src"):
            kernel[1](MockTensor(asctile.float32))

    def test_real_shape_with_l1(self, jit_test):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [256])
            asctile.copy_in(x_gm, [0], [128], asctile.TensorLocation.L1, real_shape=[64])

        with pytest.raises(RuntimeError, match="real_shape"):
            kernel[1](MockTensor(asctile.float32))

    def test_with_plain_value_offset(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress, offset: int):
            x_gm = asctile.global_tensor(x_ptr, [256])
            asctile.copy_in(x_gm, [offset], [128])

        kernel[1](MockTensor(asctile.float32), MockValue(asctile.int32))
        assert mock_launch.call_count == 1


class TestCopyOut:

    @pytest.mark.parametrize("dtype", valid_dtypes)
    def test_copy_out(self, jit_test, mock_launch, zero_tile, dtype):

        @jit_test
        def kernel(out_ptr: asctile.GlobalAddress):
            out_gm = asctile.global_tensor(out_ptr, [128])
            src = zero_tile([128], dtype)
            asctile.copy_out(src, out_gm, [0])

        kernel[1](MockTensor(dtype))
        assert mock_launch.call_count == 1

    def test_scalar(self, jit_test, mock_launch):

        @jit_test
        def kernel(out_ptr: asctile.GlobalAddress):
            out_gm = asctile.global_tensor(out_ptr, [128])
            asctile.copy_out(42.0, out_gm, [0])

        kernel[1](MockTensor(asctile.float32))
        assert mock_launch.call_count == 1

    def test_with_real_shape(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel(out_ptr: asctile.GlobalAddress):
            out_gm = asctile.global_tensor(out_ptr, [128])
            src = zero_tile([128], asctile.float32)
            asctile.copy_out(src, out_gm, [0], real_shape=[64])

        kernel[1](MockTensor(asctile.float32))
        assert mock_launch.call_count == 1

    def test_from_l0c(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel(out_ptr: asctile.GlobalAddress):
            out_gm = asctile.global_tensor(out_ptr, [128])
            src = zero_tile([128], asctile.float32, asctile.TensorLocation.L0C)
            asctile.copy_out(src, out_gm, [0])

        kernel[1](MockTensor(asctile.float32))
        assert mock_launch.call_count == 1

    def test_invalid_dst_type(self, jit_test, zero_tile):

        @jit_test
        def kernel(out_ptr: asctile.GlobalAddress):
            src = zero_tile([128], asctile.float32)
            asctile.copy_out(src, "invalid", [0])

        with pytest.raises(TypeError, match="dst"):
            kernel[1](MockTensor(asctile.float32))

    def test_with_plain_value_offset(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel(out_ptr: asctile.GlobalAddress, offset: int):
            out_gm = asctile.global_tensor(out_ptr, [256])
            src = zero_tile([128], asctile.float32)
            asctile.copy_out(src, out_gm, [offset])

        kernel[1](MockTensor(asctile.float32), MockValue(asctile.int32))
        assert mock_launch.call_count == 1


class TestCopy:

    def test_ub_to_l1(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [128])
            ub = asctile.copy_in(x_gm, [0], [128])
            asctile.copy(ub, location=asctile.TensorLocation.L1)

        kernel[1](MockTensor(asctile.float32))
        assert mock_launch.call_count == 1

    def test_l0c_to_ub(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            src = zero_tile([128], asctile.float32, asctile.TensorLocation.L0C)
            asctile.copy(src, location=asctile.TensorLocation.UB)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_l0c_to_l1(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            src = zero_tile([128], asctile.float32, asctile.TensorLocation.L0C)
            asctile.copy(src, location=asctile.TensorLocation.L1)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_l1_to_l0a(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            src = zero_tile([128], asctile.float32, asctile.TensorLocation.L1)
            asctile.copy(src, location=asctile.TensorLocation.L0A)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_l1_to_l0b(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            src = zero_tile([128], asctile.float32, asctile.TensorLocation.L1)
            asctile.copy(src, location=asctile.TensorLocation.L0B)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_l1_to_bt(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            src = zero_tile([128], asctile.float32, asctile.TensorLocation.L1)
            asctile.copy(src, location=asctile.TensorLocation.BT)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_with_offsets_and_shape(self, jit_test, mock_launch, zero_tile):

        @jit_test
        def kernel():
            src = zero_tile([64, 128], asctile.float32, asctile.TensorLocation.L1)
            asctile.copy(src, [16, 16], [32, 32], asctile.TensorLocation.L0A)

        kernel[1]()
        assert mock_launch.call_count == 1

    @pytest.mark.parametrize("distrib, expected_shape", (
        (asctile.DistribMode.FullVec0, (32, 64)),
        (asctile.DistribMode.FullVec1, (32, 64)),
        (asctile.DistribMode.SplitByM, (16, 64)),
        (asctile.DistribMode.SplitByN, (32, 32)),
    ))
    def test_with_distrib(self, jit_test, mock_launch, zero_tile, distrib, expected_shape):

        @jit_test
        def kernel():
            src = zero_tile([32, 64], asctile.float32, asctile.TensorLocation.L0C)
            result = asctile.copy(src, location=asctile.TensorLocation.UB, distrib=distrib)
            asctile.static_assert(result.shape == expected_shape)

        kernel[1]()
        assert mock_launch.call_count == 1

    def test_distrib_wrong_rank(self, jit_test, zero_tile):

        @jit_test
        def kernel():
            src = zero_tile([32], asctile.float32, asctile.TensorLocation.L0C)
            asctile.copy(src, location=asctile.TensorLocation.UB, distrib=asctile.DistribMode.SplitByM)

        with pytest.raises(RuntimeError, match="2D"):
            kernel[1]()

    @pytest.mark.parametrize("distrib, wrong_shape", (
        (asctile.DistribMode.SplitByM, (32, 64)),
        (asctile.DistribMode.SplitByM, (16, 32)),
        (asctile.DistribMode.SplitByN, (16, 32)),
    ))
    def test_distrib_wrong_shape(self, jit_test, zero_tile, distrib, wrong_shape):

        @jit_test
        def kernel():
            src = zero_tile([32, 64], asctile.float32, asctile.TensorLocation.L0C)
            asctile.copy(src, [0, 0], wrong_shape, asctile.TensorLocation.UB, distrib)

        with pytest.raises(RuntimeError, match="split shape"):
            kernel[1]()

    @pytest.mark.parametrize("shape, distrib", (
        ([31, 64], asctile.DistribMode.SplitByM),
        ([32, 31], asctile.DistribMode.SplitByN),
    ))
    def test_distrib_unaligned_shape(self, jit_test, zero_tile, shape, distrib):

        @jit_test
        def kernel():
            src = zero_tile(shape, asctile.float32, asctile.TensorLocation.L0C)
            asctile.copy(src, location=asctile.TensorLocation.UB, distrib=distrib)

        with pytest.raises(RuntimeError, match="multiple of"):
            kernel[1]()

    @pytest.mark.parametrize("distrib, expected_shape", (
        (asctile.DistribMode.JoinByM, (32, 64)),
        (asctile.DistribMode.JoinByN, (16, 128)),
    ))
    def test_with_join_distrib(self, jit_test, mock_launch, zero_tile, distrib, expected_shape):

        @jit_test
        def kernel():
            src = zero_tile([16, 64], asctile.float32, asctile.TensorLocation.UB)
            result = asctile.copy(src, location=asctile.TensorLocation.L1, distrib=distrib)
            asctile.static_assert(result.shape == expected_shape)

        kernel[1]()
        assert mock_launch.call_count == 1

    @pytest.mark.parametrize("distrib, expected_shape", (
        (asctile.DistribMode.JoinByM, (32, 64)),
        (asctile.DistribMode.JoinByN, (16, 128)),
    ))
    def test_join_distrib_explicit_shape(self, jit_test, mock_launch, zero_tile, distrib, expected_shape):

        @jit_test
        def kernel():
            src = zero_tile([16, 64], asctile.float32, asctile.TensorLocation.UB)
            result = asctile.copy(src, [0, 0], expected_shape, asctile.TensorLocation.L1, distrib)
            asctile.static_assert(result.shape == expected_shape)

        kernel[1]()
        assert mock_launch.call_count == 1

    @pytest.mark.parametrize("distrib, wrong_shape", (
        (asctile.DistribMode.JoinByM, (16, 64)),
        (asctile.DistribMode.JoinByN, (16, 64)),
        (asctile.DistribMode.JoinByM, (64, 64)),
    ))
    def test_join_distrib_wrong_shape(self, jit_test, zero_tile, distrib, wrong_shape):

        @jit_test
        def kernel():
            src = zero_tile([16, 64], asctile.float32, asctile.TensorLocation.UB)
            asctile.copy(src, [0, 0], wrong_shape, asctile.TensorLocation.L1, distrib)

        with pytest.raises(RuntimeError, match="joined shape"):
            kernel[1]()


class TestTo:

    def test_to_location(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [128])
            ub = asctile.copy_in(x_gm, [0], [128])
            ub.to(asctile.TensorLocation.L1)

        kernel[1](MockTensor(asctile.float32))
        assert mock_launch.call_count == 1

    def test_to_dtype(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [128])
            ub = asctile.copy_in(x_gm, [0], [128])
            ub.to(asctile.float16)

        kernel[1](MockTensor(asctile.float32))
        assert mock_launch.call_count == 1

    def test_to_same_location(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [128])
            ub = asctile.copy_in(x_gm, [0], [128])
            ub.to(asctile.TensorLocation.UB)

        kernel[1](MockTensor(asctile.float32))
        assert mock_launch.call_count == 1


class TestGather:

    @pytest.mark.parametrize("dtype", valid_dtypes)
    @pytest.mark.parametrize("index_dtype", index_dtypes)
    @pytest.mark.parametrize("check_bounds", (True, False))
    def test_gather(self, jit_test, mock_launch, dtype, index_dtype, check_bounds):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress, check_bounds: asctile.ConstExpr):
            x_gm = asctile.global_tensor(x_ptr, [64, 128])
            index = asctile.full([64], 0, index_dtype)
            asctile.gather(x_gm, [0], 0, index, check_bounds=check_bounds)

        kernel[1](MockTensor(dtype), check_bounds)
        assert mock_launch.call_count == 1

    def test_gather_wrong_index(self, jit_test):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [64, 128])
            index = asctile.full([2, 64], 0, asctile.int32)
            asctile.gather(x_gm, [0], 0, index, check_bounds=True)

        with pytest.raises(ValueError, match="index"):
            kernel[1](MockTensor(asctile.int32))

    def test_gather_wrong_dim1(self, jit_test):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [64, 128])
            index = asctile.full([64], 0, asctile.int32)
            asctile.gather(x_gm, [0, 0, 0], 2, index, check_bounds=True)

        with pytest.raises(ValueError, match="dim"):
            kernel[1](MockTensor(asctile.int32))

    def test_gather_wrong_dynamic(self, jit_test):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress, size):
            x_gm = asctile.global_tensor(x_ptr, [64, size])
            index = asctile.full([64], 0, asctile.int32)
            asctile.gather(x_gm, [0], 0, index, check_bounds=True)

        with pytest.raises(ValueError, match="src"):
            kernel[1](MockTensor(asctile.int32), 128)

    def test_gather_last_dim(self, jit_test):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [64, 128])
            index = asctile.full([64], 0, asctile.int32)
            asctile.gather(x_gm, [0, 0], 1, index, check_bounds=True)

        with pytest.raises(NotImplementedError, match="not implemented"):
            kernel[1](MockTensor(asctile.int32))

    def test_gather_with_options(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [64, 128])
            index = asctile.full([64], 0, asctile.int32)
            asctile.gather(x_gm, [0], 0, index, num_indices=32, pad_value=0.0, check_bounds=True)

        kernel[1](MockTensor(asctile.float32))
        assert mock_launch.call_count == 1


class TestScatter:

    @pytest.mark.parametrize("dtype", valid_dtypes)
    @pytest.mark.parametrize("index_dtype", index_dtypes)
    @pytest.mark.parametrize("check_bounds", (True, False))
    def test_scatter(self, jit_test, mock_launch, dtype, index_dtype, check_bounds):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress, check_bounds: asctile.ConstExpr):
            x_gm = asctile.global_tensor(x_ptr, [64, 128])
            index = asctile.full([64], 0, index_dtype)
            data = asctile.full([64, 128], 0, dtype)
            asctile.scatter(data, 0, index, x_gm, [0], check_bounds=check_bounds)

        kernel[1](MockTensor(dtype), check_bounds)
        assert mock_launch.call_count == 1

    def test_scatter_wrong_dynamic(self, jit_test):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress, size):
            x_gm = asctile.global_tensor(x_ptr, [64, size])
            index = asctile.full([64], 0, asctile.int32)
            data = asctile.full([64, 128], 0, asctile.float32)
            asctile.scatter(data, 0, index, x_gm, [0], check_bounds=True)

        with pytest.raises(ValueError, match="dst"):
            kernel[1](MockTensor(asctile.float32), 128)

    def test_scatter_type_mismatch(self, jit_test):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [64, 128])
            index = asctile.full([64], 0, asctile.int32)
            data = asctile.full([64, 128], 0, asctile.float16)
            asctile.scatter(data, 0, index, x_gm, [0], check_bounds=True)

        with pytest.raises(ValueError, match="data types"):
            kernel[1](MockTensor(asctile.float32))

    def test_scatter_shape_mismatch(self, jit_test):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [64, 64, 128])
            index = asctile.full([64], 0, asctile.int32)
            data = asctile.full([32, 128], 0, asctile.float32)
            asctile.scatter(data, 0, index, x_gm, [0], check_bounds=True)

        with pytest.raises(ValueError, match="src"):
            kernel[1](MockTensor(asctile.float32))

    def test_scatter_wrong_dim(self, jit_test):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [64, 128])
            index = asctile.full([64], 0, asctile.int32)
            data = asctile.full([64, 128], 0, asctile.float32)
            asctile.scatter(data, 2, index, x_gm, [0, 0, 0], check_bounds=True)

        with pytest.raises(ValueError, match="dim"):
            kernel[1](MockTensor(asctile.float32))

    def test_scatter_last_dim(self, jit_test):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [64, 128])
            index = asctile.full([64], 0, asctile.int32)
            data = asctile.full([128], 0, asctile.float32)
            asctile.scatter(data, 1, index, x_gm, [0, 0], check_bounds=True)

        with pytest.raises(NotImplementedError, match="not implemented"):
            kernel[1](MockTensor(asctile.float32))

    def test_scatter_wrong_index(self, jit_test):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [64, 128])
            index = asctile.full([2, 64], 0, asctile.int32)
            data = asctile.full([64, 128], 0, asctile.float32)
            asctile.scatter(data, 0, index, x_gm, [0], check_bounds=True)

        with pytest.raises(ValueError, match="index"):
            kernel[1](MockTensor(asctile.float32))

    def test_scatter_dim_size_mismatch(self, jit_test):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [64, 64, 128])
            index = asctile.full([64], 0, asctile.int32)
            data = asctile.full([64, 32, 128], 0, asctile.float32)
            asctile.scatter(data, 0, index, x_gm, [0], check_bounds=True)

        with pytest.raises(ValueError, match="src"):
            kernel[1](MockTensor(asctile.float32))

    def test_scatter_num_indices(self, jit_test, mock_launch):

        @jit_test
        def kernel(x_ptr: asctile.GlobalAddress):
            x_gm = asctile.global_tensor(x_ptr, [64, 128])
            index = asctile.full([64], 0, asctile.int32)
            data = asctile.full([64, 128], 0, asctile.float32)
            asctile.scatter(data, 0, index, x_gm, [0], num_indices=32, check_bounds=True)

        kernel[1](MockTensor(asctile.float32))
        assert mock_launch.call_count == 1
