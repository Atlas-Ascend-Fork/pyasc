# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from asc.experimental import asctile
import pytest


def test_block_idx(jit_test, mock_launch):

    @jit_test
    def kernel():
        idx = asctile.block_idx()
        asctile.static_assert(idx.dtype == asctile.int32)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_block_num(jit_test, mock_launch):

    @jit_test
    def kernel():
        num = asctile.block_num()
        asctile.static_assert(num.dtype == asctile.int32)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_sub_block_idx(jit_test, mock_launch):

    @jit_test
    def kernel():
        idx = asctile.sub_block_idx()
        asctile.static_assert(idx.dtype == asctile.int32)

    kernel[1]()
    assert mock_launch.call_count == 1


def test_sub_block_num(jit_test, mock_launch):

    @jit_test
    def kernel():
        num = asctile.sub_block_num()
        asctile.static_assert(num.dtype == asctile.int32)

    kernel[1]()
    assert mock_launch.call_count == 1


@pytest.mark.parametrize("split", (asctile.DistribMode.SplitByM, asctile.DistribMode.SplitByN))
def test_cv_strategy_split_by_axis(split, jit_test, mock_launch):

    @jit_test
    def kernel():
        with asctile.cv_strategy(split):
            pass

    kernel[1]()
    assert mock_launch.call_count == 1


@pytest.mark.parametrize("split", (asctile.DistribMode.FullVec0, asctile.DistribMode.FullVec1))
def test_cv_strategy_split_by_aiv(split, jit_test):

    @jit_test
    def kernel():
        with asctile.cv_strategy(split):
            pass

    with pytest.raises(ValueError, match="Splitting by axis must be requested"):
        kernel[1]()
