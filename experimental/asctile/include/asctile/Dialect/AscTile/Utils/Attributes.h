/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_DIALECT_ASCTILE_UTILS_ATTRIBUTES_H
#define ASCTILE_DIALECT_ASCTILE_UTILS_ATTRIBUTES_H
#define LITERAL constexpr const char*

namespace mlir {
namespace asctile {
namespace attr {

LITERAL gmBarrier = "asctile.gm_barrier";
LITERAL hasBias = "asctile.has_bias";
LITERAL isBias = "asctile.is_bias";
LITERAL isMatrixA = "asctile.is_matrix_a";
LITERAL needSplit = "asctile.need_split";
LITERAL reuseSource = "asctile.reuse_source";
LITERAL splitShape = "asctile.split_shape";
LITERAL transposeAL0 = "asctile.transpose_a_l0";
LITERAL transposeAL1 = "asctile.transpose_a_l1";
LITERAL transposeBL0 = "asctile.transpose_b_l0";
LITERAL transposeBL1 = "asctile.transpose_b_l1";
LITERAL transposeDims = "asctile.transpose_dims";
LITERAL unrollFactor = "asctile.unroll_factor";
LITERAL unrolledLoop = "asctile.unrolled_loop";
LITERAL unrollIter = "asctile.unroll_iter";

} // namespace attr
} // namespace asctile
} // namespace mlir

#undef LITERAL
#endif // ASCTILE_DIALECT_ASCTILE_UTILS_ATTRIBUTES_H
