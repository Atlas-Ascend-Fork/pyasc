# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass
from typing import Literal, Optional

from asc._C import ir, passes
from asc._C.libpyasc import asctile
from asc.runtime.compiler import CompileOptions as CompileOptionsBase, Compiler as CompilerBase, CompilationArch


@dataclass
class CompileOptions(CompileOptionsBase):
    """Binary compilation and IR transformation options (for ``asctile`` kernels)"""

    cv_ratio: Literal[1, 2] = 1
    """
    The ratio between Vector and Cube units. For example, ``Ascend950PR_9599`` cores have two vector sub-cores per each
    cube sub-core, hence the supported ratio values are 1 and 2.
    """

    debug = False
    """
    Enable debug mode for the kernel.

    When ``True``, device-side debug operations (``device_print``, ``device_assert``) are active during
    kernel execution. When ``False`` (the default), these operations are removed from the compiled kernel.

    Use this option to enable debugging output and assertions on the device. Note that debug operations
    may impact performance and should be disabled in production builds.
    """

    insert_sync: bool = True
    """
    Insert synchronization instructions automatically.
    **This feature is enabled by default**, which is usually a must, but may be disabled for the debugging purposes.
    """

    reuse_alloc: Literal[0, 1, 2] = 0
    """
    Try to reduce the on-chip memory usage by replacing the tensors with those allocated earlier but became unused.
    Having this feature enabled may help to avoid memory overflow but may introduce performance regressions.

    ===== ======
    Value Effect
    ===== ======
    ``0`` Disable the feature (default)
    ``1`` Enable the feature, use a legacy implementation
    ``2`` Enable the feature, use a modern implementation (recommended)
    ===== ======
    """

    static_alloc: Optional[bool] = None
    """
    Perform static allocation for tiles instead of relying on Ascend C TPipe backend.
    The static allocation feature may help to reduce an overhead caused by scalar code.

    **This feature is enabled by default** on supported platforms (such as ``Ascend950PR_9599``).
    """

    vf_fusion: bool = False
    """
    Fuse groups of consecutive vector operations into VF blocks using Ascend C register API.
    This feature may help to eliminate unnecessary memory transfers and improve data locality.
    """


class Compiler(CompilerBase):
    options_cls = CompileOptions

    def __init__(self, options: Optional[CompileOptions] = None):
        super().__init__(options)
        if self.options.cv_ratio not in (1, 2):
            raise RuntimeError(f"'cv_ratio' is only allowed to be 1, 2 for the {self.arch} architecture")
        if self.options.reuse_alloc not in (0, 1, 2):
            raise RuntimeError("'reuse_alloc' is only allowed to be 0, 1, 2")
        if self.options.vf_fusion and self.arch != CompilationArch.C310:
            raise RuntimeError(f"VF fusion is not supported for the {self.arch} architecture")

    def preprocess_module(self, mod: ir.ModuleOp) -> None:
        super().preprocess_module(mod)
        builder = ir.Builder(mod.op)
        mod.set_attr(asctile.ir.attr.cv_ratio, builder.get_i64_attr(self.options.cv_ratio))
        if self.options.static_alloc is not None:
            mod.set_attr(asctile.ir.attr.static_alloc, builder.get_bool_attr(self.options.static_alloc))

    def postprocess_module(self, mod: ir.ModuleOp) -> None:
        super().postprocess_module(mod)
        self.enable_debug = self.options.debug

    def schedule_passes(self, pm: passes.PassManager) -> None:
        arch_c310 = self.arch == CompilationArch.C310
        passes.ascendc.add_privatize_func(pm)
        passes.common.add_inliner(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_reconcile_unrealized_casts(pm)
        asctile.passes.asctile.add_resolve_auto_location(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        asctile.passes.asctile.add_location_cast_to_copy(pm)
        asctile.passes.asctile.add_fulfill_data_transfer(pm)
        asctile.passes.asctile.add_verify_tensor_location(pm)
        asctile.passes.asctile.add_split_cube_load(pm)
        asctile.passes.asctile.add_prepare_cv_strategy(pm)
        asctile.passes.asctile.add_apply_cv_strategy(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        asctile.passes.asctile.add_combine_extract_slice(pm)
        asctile.passes.asctile.add_cube_transpose_to_load(pm)
        asctile.passes.asctile.add_legalize_matmul(pm)
        passes.common.add_canonicalizer(pm)
        asctile.passes.asctile.add_mark_matmul_acc_with_bias(pm)
        asctile.passes.asctile.add_apply_homomorphism(pm)
        asctile.passes.asctile.add_fold_cast(pm)
        asctile.passes.asctile.add_transform_math_ops(pm)
        asctile.passes.asctile.add_transform_store_fixpipe(pm)
        asctile.passes.asctile.add_detect_bias_load(pm)
        asctile.passes.asctile.add_mark_reuse_source(pm)
        if arch_c310:
            asctile.passes.asctile.add_vector_transpose_to_load_store(pm)
            asctile.passes.asctile.add_unscalarize_reduction(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
        asctile.passes.asctile.add_wrap_cv_groups(pm)
        asctile.passes.asctile.add_merge_cv_groups(pm)
        if not self.options.debug:
            asctile.passes.ascendc.add_remove_debug_ops(pm)
        asctile.passes.asclower.add_expand_math(pm)
        asctile.passes.asclower.add_redress_i1_tensor(pm)
        asctile.passes.asclower.add_lower_arith(pm)
        asctile.passes.asclower.add_lower_arith_binary(pm)
        asctile.passes.asclower.add_lower_atomic(pm)
        asctile.passes.asclower.add_lower_asctile_data_transfer(pm)
        asctile.passes.asclower.add_lower_asctile(pm)
        asctile.passes.asclower.add_lower_asctile_to_basic(pm)
        asctile.passes.asclower.add_lower_math(pm)
        asctile.passes.asclower.add_lower_scf(pm)
        asctile.passes.asclower.add_lower_tensor(pm)
        asctile.passes.asclower.add_displace_concat(pm)
        passes.common.add_canonicalizer(pm)
        asctile.passes.asclower.add_realize_conversion_cast(pm)
        asctile.passes.asclower.add_expand_mask(pm)
        asctile.passes.ascendc.add_promote_cv_block(pm)
        asctile.passes.ascendc.add_fill_asc_operands(pm)
        asctile.passes.ascendc.add_fixup_mmad_acc_params_pass(pm)
        passes.ascendc.add_input_output_tensor(pm)
        if self.options.reuse_alloc == 1:
            asctile.passes.ascendc.add_reuse_ub_allocation(pm, reuse_in_out=True)
        asctile.passes.asctile.add_unroll_loop(pm, annotate=True)
        asctile.passes.ascendc.add_compute_reuse_group(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.ascendc.add_hoist_tensor_allocation(pm, exclude_in_out=not arch_c310)
        asctile.passes.ascendc.add_refine_cube_position(pm)
        if self.options.reuse_alloc == 1:
            asctile.passes.ascendc.add_reuse_ub_allocation(pm, reuse_in_out=False)
        elif self.options.reuse_alloc == 2:
            asctile.passes.ascendc.add_reuse_tensor_allocation(pm)
        passes.common.add_canonicalizer(pm)
        if self.options.vf_fusion:
            asctile.passes.ascvf.add_promote_cast_ops(pm)
            asctile.passes.ascvf.add_find_vf_group(pm)
            asctile.passes.ascvf.add_lower_to_reg(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
            asctile.passes.ascvf.add_dispatch_vf_fusion(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
            asctile.passes.ascvf.add_materialize_load_store(pm)
        asctile.passes.ascendc.add_dispatch_alloc(pm)
        asctile.passes.ascendc.add_unify_bias_tensor(pm)
        passes.ascendc.add_unify_pipe(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_sccp(pm)
        passes.common.add_canonicalizer(pm)
        asctile.passes.ascendc.add_promote_cv_block(pm)
        asctile.passes.ascendc.add_insert_cross_core_sync(pm)
        asctile.passes.ascendc.add_insert_cross_core_sync_gm(pm)
        if self.options.insert_sync:
            passes.ascendc.add_erase_sync(pm)
            passes.ascendc.add_hoist_que_bind(pm)
            if arch_c310:
                asctile.passes.ascendc.add_insert_bufid_sync(pm)
                asctile.passes.ascendc.add_insert_bias_bufid_sync(pm)
                passes.common.add_canonicalizer(pm)
                asctile.passes.ascendc.add_fuse_bufid_sync(pm)
            else:
                passes.ascendc.add_insert_que_sync(pm)
            passes.ascendc.add_unify_pipe(pm)
            passes.common.add_canonicalizer(pm)
        asctile.passes.ascvf.add_inline_vf_group(pm)
        passes.ascendc.add_declare_py_struct(pm)
        passes.ascendc.add_generate_boilerplate(pm)
        if self.options.cv_ratio == 1:
            asctile.passes.ascendc.add_insert_subblock_guard(pm)
        if self.options.matmul_cube_only:
            passes.ascendc.add_define_cube_only(pm)
        passes.ascendc.add_legalize_kernel_args(pm, set_ffts_addr=not arch_c310)
        passes.ascendc.add_detect_kernel_type(pm)
        asctile.passes.ascendc.add_insert_init_dump(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        if self.options.verify_sync:
            passes.ascendc.add_verify_sync(pm)
        if self.options.strip_loc:
            passes.common.add_strip_debug_info(pm)
        asctile.passes.ascendc.add_compute_memory_consumption(pm)

    def _gen_init_dump_code(self, source: str, func_name: str) -> str:
        return source
