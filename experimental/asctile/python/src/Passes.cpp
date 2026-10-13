/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscendC/Transforms/Passes.h"

#include "asctile/Conversion/LowerToAsc/Passes.h"
#include "asctile/Dialect/AscTile/Transforms/Passes.h"
#include "asctile/Dialect/AscVF/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"

#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // automatic casts between containers and python types

#include "InitFuncDef.h"

#define DEFINE_ADD_PASS(NAME, CONSTRUCTOR) m.def(NAME, [](PassManager& pm) { pm.addPass(CONSTRUCTOR()); })

#define DEFINE_ADD_PASS_ON(NEST, NAME, CONSTRUCTOR) \
    m.def(NAME, [](PassManager& pm) { pm.addNestedPass<NEST>(CONSTRUCTOR()); })

namespace py = pybind11;
using namespace mlir;

namespace {

void defineAscendCPasses(py::module& mod)
{
    using namespace ascendc;
    using namespace pybind11::literals;
    auto m = mod.def_submodule("ascendc");
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_allocate_tensor", createAllocateTensorPass);
    DEFINE_ADD_PASS("add_compute_memory_consumption", createComputeMemoryConsumptionPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_compute_reuse_group", createComputeReuseGroupPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_dispatch_alloc", createDispatchAllocPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_fill_asc_operands", createFillAscOperandsPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_fixup_mmad_acc_params_pass", createFixupMmadAccParamsPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_fuse_bufid_sync", createFuseBufIdSyncPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_insert_bias_bufid_sync", createInsertBiasBufIdSyncPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_insert_bufid_sync", createInsertBufIdSyncPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_insert_cross_core_sync", createInsertCrossCoreSyncPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_insert_cross_core_sync_gm", createInsertCrossCoreSyncGMPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_insert_init_dump", createInsertInitDumpPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_insert_subblock_guard", createInsertSubBlockGuardPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_lower_to_l0", createLowerToL0Pass);
    DEFINE_ADD_PASS("add_promote_cv_block", createPromoteCVBlockPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_refine_cube_position", createRefineCubePositionPass);
    DEFINE_ADD_PASS("add_remove_debug_ops", createRemoveDebugOpsPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_reuse_tensor_allocation", createReuseTensorAllocationPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_unify_bias_tensor", createUnifyBiasTensorPass);

    m.def(
        "add_reuse_ub_allocation",
        [](PassManager& pm, bool reuseInOut) {
            pm.addNestedPass<func::FuncOp>(createReuseUBAllocationPass(reuseInOut));
        },
        "pm"_a, "reuse_in_out"_a = false);
}

void defineAscTilePasses(py::module& mod)
{
    using namespace asctile;
    using namespace pybind11::literals;
    auto m = mod.def_submodule("asctile");
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_apply_cv_strategy", createApplyCVStrategyPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_apply_homomorphism", createApplyHomomorphismPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_combine_extract_slice", createCombineExtractSlicePass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_cube_transpose_to_load", createCubeTransposeToLoadPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_detect_bias_load", createDetectBiasLoadPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_fold_cast", createFoldCastPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_fulfill_data_transfer", createFulfillDataTransferPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_legalize_matmul", createLegalizeMatmulPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_location_cast_to_copy", createLocationCastToCopyPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_mark_matmul_acc_with_bias", createMarkMatmulAccWithBiasPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_mark_reuse_source", createMarkReuseSourcePass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_merge_cv_groups", createMergeCVGroupsPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_prepare_cv_strategy", createPrepareCVStrategyPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_promote_pure_operations", createPromotePureOpsPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_resolve_auto_location", createResolveAutoLocationPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_split_cube_load", createSplitCubeLoadPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_transform_math_ops", createTransformMathOpsPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_transform_store_fixpipe", createTransformStoreFixpipePass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_unscalarize_reduction", createUnscalarizeReductionPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_vector_transpose_to_load_store", createVectorTransposeToLoadStorePass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_verify_tensor_location", createVerifyTensorLocationPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_wrap_cv_groups", createWrapCVGroupsPass);

    m.def(
        "add_unroll_loop",
        [](PassManager& pm, bool annotate) { pm.addNestedPass<func::FuncOp>(createUnrollLoopPass(annotate)); }, "pm"_a,
        "annotate"_a = false);
}

void defineAscVFPasses(py::module& mod)
{
    using namespace ascvf;
    using namespace pybind11::literals;
    auto m = mod.def_submodule("ascvf");
    DEFINE_ADD_PASS("add_dispatch_vf_fusion", createDispatchVFFusionPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_eliminate_common_mask", createEliminateCommonMaskPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_erase_barrier", createEraseBarrierPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_eliminate_data_transfer", createEliminateDataTransferPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_find_vf_group", createFindVFGroupPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_fuse_vf_for", createFuseVFForPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_inline_vf_group", createInlineVFGroupPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_lower_to_reg", createLowerToRegPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_materialize_load_store", createMaterializeLoadStorePass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_promote_cast_ops", createPromoteCastOpsPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_reorder_ops_in_vec_scope", createReorderOpsInVecScopePass);
}

void defineLowerToAscPasses(py::module& mod)
{
    using namespace asclower;
    auto m = mod.def_submodule("asclower");
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_displace_concat", createDisplaceConcatPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_expand_mask", createExpandMaskPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_expand_math", createExpandMathPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_lower_arith", createLowerArithPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_lower_arith_binary", createLowerArithBinaryPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_lower_atomic", createLowerAtomicPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_lower_asctile", createLowerAscTilePass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_lower_asctile_data_transfer", createLowerAscTileDataTransferPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_lower_asctile_to_basic", createLowerAscTileToBasicPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_lower_math", createLowerMathPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_lower_scf", createLowerSCFPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_lower_tensor", createLowerTensorPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_realize_conversion_cast", createRealizeConversionCastPass);
    DEFINE_ADD_PASS_ON(func::FuncOp, "add_redress_i1_tensor", createRedressI1TensorPass);
}

} // namespace

void mlir::asctile::initPassesModule(py::module&& m)
{
    defineAscendCPasses(m);
    defineAscTilePasses(m);
    defineAscVFPasses(m);
    defineLowerToAscPasses(m);
}

#undef DEFINE_ADD_PASS
#undef DEFINE_ADD_PASS_ON
