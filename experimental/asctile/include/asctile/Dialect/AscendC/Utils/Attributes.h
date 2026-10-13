/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_DIALECT_ASCENDC_UTILS_ATTRIBUTES_H
#define ASCTILE_DIALECT_ASCENDC_UTILS_ATTRIBUTES_H
#define LITERAL constexpr const char*

namespace mlir {
namespace ascendc {

namespace attr {
LITERAL bufId = "ascendc.buf_id";
LITERAL bufIds = "ascendc.buf_ids";
LITERAL calCountSet = "asc.cal_count_set";
LITERAL crossCoreFlagId = "ascendc.cross_core_flag_id";
LITERAL cvRatio = "ascendc.cv_ratio";
LITERAL maskSet = "asc.mask_set";
LITERAL reuseGroup = "asc.reuse_group";
LITERAL staticAlloc = "asc.static_alloc";
} // namespace attr

} // namespace ascendc
} // namespace mlir

#undef LITERAL
#endif // ASCTILE_DIALECT_ASCENDC_UTILS_ATTRIBUTES_H
