// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_data_copy_vld_reg(AscendC::Reg::RegTensor<float> v1, float* v2) {
// CHECK-NEXT:  AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(v1, v2);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_data_copy_vld_reg(%dstReg: !ascendc.reg_tensor<f32>, %src: memref<?xf32>) {
  ascendc.data_copy_vld_reg %dstReg, %src {dist = 0 : i32} : !ascendc.reg_tensor<f32>, memref<?xf32>
  return
}

// CHECK-LABEL:void emit_data_copy_vst_reg(float* v1, AscendC::Reg::RegTensor<float> v2, AscendC::Reg::MaskReg v3) {
// CHECK-NEXT:  AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(v1, v2, v3);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_data_copy_vst_reg(%dst: memref<?xf32>, %srcReg: !ascendc.reg_tensor<f32>, %maskReg: !ascendc.mask_reg) {
  ascendc.data_copy_vst_reg %dst, %srcReg, %maskReg {dist = 16 : i32} : memref<?xf32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  return
}

// CHECK-LABEL:void emit_vld_unalign_reg(AscendC::Reg::RegTensor<float> v1, AscendC::Reg::UnalignReg v2, float* v3) {
// CHECK-NEXT:  AscendC::Reg::LoadUnAlign(v1, v2, v3);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_vld_unalign_reg(%dstReg: !ascendc.reg_tensor<f32>, %uReg: !ascendc.unalign_reg, %src: memref<?xf32>) {
  ascendc.vld_unalign_reg %dstReg, %uReg, %src : !ascendc.reg_tensor<f32>, !ascendc.unalign_reg, memref<?xf32>
  return
}

// CHECK-LABEL:void emit_vld_unalign_pre_reg(AscendC::Reg::UnalignReg v1, float* v2) {
// CHECK-NEXT:  AscendC::Reg::LoadUnAlignPre(v1, v2);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_vld_unalign_pre_reg(%uReg: !ascendc.unalign_reg, %src: memref<?xf32>) {
  ascendc.vld_unalign_pre_reg %uReg, %src : !ascendc.unalign_reg, memref<?xf32>
  return
}

// CHECK-LABEL:void emit_vst_unalign_reg(float* v1, AscendC::Reg::RegTensor<float> v2, AscendC::Reg::UnalignReg v3) {
// CHECK-NEXT:  AscendC::Reg::StoreUnAlign(v1, v2, v3);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_vst_unalign_reg(%dst: memref<?xf32>, %srcReg: !ascendc.reg_tensor<f32>, %uReg: !ascendc.unalign_reg) {
  ascendc.vst_unalign_reg %dst, %srcReg, %uReg : memref<?xf32>, !ascendc.reg_tensor<f32>, !ascendc.unalign_reg
  return
}

// CHECK-LABEL:void emit_vst_unalign_post_reg(float* v1, AscendC::Reg::UnalignReg v2) {
// CHECK-NEXT:  AscendC::Reg::StoreUnAlignPost(v1, v2);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_vst_unalign_post_reg(%dst: memref<?xf32>, %uReg: !ascendc.unalign_reg) {
  ascendc.vst_unalign_post_reg %dst, %uReg : memref<?xf32>, !ascendc.unalign_reg
  return
}

// CHECK-LABEL:void emit_mask_gen_with_reg_tensor(AscendC::Reg::MaskReg v1, AscendC::Reg::RegTensor<uint8_t> v2) {
// CHECK-NEXT:  AscendC::Reg::MaskGenWithRegTensor<uint32_t, 0>(v1, reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(v2));
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_mask_gen_with_reg_tensor(%dst: !ascendc.mask_reg, %src: !ascendc.reg_tensor<ui8>) {
  ascendc.mask_gen_with_reg_tensor %dst, %src {type = ui32} : !ascendc.mask_reg, !ascendc.reg_tensor<ui8>
  return
}

// CHECK-LABEL:void emit_compare_scalar_reg(AscendC::Reg::MaskReg v1, AscendC::Reg::RegTensor<float> v2, float v3, AscendC::Reg::MaskReg v4) {
// CHECK-NEXT:  AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::EQ>(v1, v2, v3, v4);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_compare_scalar_reg(%dst: !ascendc.mask_reg, %src: !ascendc.reg_tensor<f32>, %cst: f32, %mask: !ascendc.mask_reg) {
  ascendc.compare_scalar_reg %dst, %src, %cst, %mask {cmpMode = 2 : i64} : !ascendc.mask_reg, !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  return
}

// CHECK-LABEL:void emit_update_mask(int32_t* v1) {
// CHECK:       AscendC::Reg::MaskReg v2 = AscendC::Reg::UpdateMask<float>(*v1);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_update_mask(%count: memref<?xi32>) {
  %mask = ascendc.update_mask f32, %count : memref<?xi32>
  return
}

// CHECK-LABEL:void emit_reg_tensor() {
// CHECK-NEXT:  AscendC::Reg::RegTensor<float> v1;
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_reg_tensor() {
  %reg = ascendc.reg_tensor : !ascendc.reg_tensor<f32>
  return
}

// CHECK-LABEL:void emit_unalign_reg() {
// CHECK-NEXT:  AscendC::Reg::UnalignReg v1 = AscendC::Reg::UnalignReg();
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_unalign_reg() {
  %reg = ascendc.unalign_reg : !ascendc.unalign_reg
  return
}

// CHECK-LABEL:void emit_mask_reg() {
// CHECK-NEXT:  AscendC::Reg::MaskReg v1 = AscendC::Reg::MaskReg();
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_mask_reg() {
  %mask = ascendc.mask_reg : !ascendc.mask_reg
  return
}

// CHECK-LABEL:void emit_duplicate_scalar_reg(AscendC::Reg::RegTensor<float> v1, float v2) {
// CHECK-NEXT:  Duplicate(v1, v2);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_duplicate_scalar_reg(%dstReg: !ascendc.reg_tensor<f32>, %scalar: f32) {
  ascendc.duplicate %dstReg, %scalar : !ascendc.reg_tensor<f32>, f32
  return
}

// CHECK-LABEL:void emit_get_vec_len() {
// CHECK-NEXT:  int32_t v1 = AscendC::GetVecLen();
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_get_vec_len() {
  %len = ascendc.get_vec_len : i32
  return
}

// CHECK-LABEL:void emit_local_mem_bar() {
// CHECK-NEXT:  AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_local_mem_bar() {
  ascendc.local_mem_bar VEC_STORE, VEC_LOAD
  return
}

// CHECK-LABEL:void emit_select_reg(AscendC::Reg::RegTensor<float> v1, AscendC::Reg::RegTensor<float> v2, AscendC::Reg::RegTensor<float> v3, AscendC::Reg::MaskReg v4) {
// CHECK-NEXT:  AscendC::Reg::Select(v1, v2, v3, v4);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_select_reg(%dstReg: !ascendc.reg_tensor<f32>, %src0Reg: !ascendc.reg_tensor<f32>, %src1Reg: !ascendc.reg_tensor<f32>, %maskReg: !ascendc.mask_reg) {
  ascendc.select_reg %dstReg, %src0Reg, %src1Reg, %maskReg : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  return
}

// CHECK-LABEL:void emit_create_mask() {
// CHECK-NEXT:  AscendC::Reg::MaskReg v1 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_create_mask() {
  %mask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
  return
}

// CHECK-LABEL:void emit_binary_reg(AscendC::Reg::RegTensor<float> v1, AscendC::Reg::RegTensor<float> v2, AscendC::Reg::RegTensor<float> v3, AscendC::Reg::MaskReg v4) {
// CHECK-NEXT:  AscendC::Reg::Add(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::And(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::Div(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::FusedAbsSub(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::FusedExpSub(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::FusedMulDstAdd(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::Sub(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::Max(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::Min(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::Mul(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::MulAddDst(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::Or(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::Prelu(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::Xor(v1, v2, v3, v4);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_binary_reg(%dst: !ascendc.reg_tensor<f32>, %src0: !ascendc.reg_tensor<f32>, %src1: !ascendc.reg_tensor<f32>, %mask: !ascendc.mask_reg) {
  ascendc.add_reg %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.and_reg %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.div_reg %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.fused_abs_sub_reg %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.fused_exp_sub_reg %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.fused_mul_dst_add_reg %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.sub_reg %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.max_reg %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.min_reg %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.mul_reg %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.mul_add_dst_reg %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.or_reg %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.prelu_reg %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.xor_reg %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  return
}

// CHECK-LABEL:void emit_unary_reg(AscendC::Reg::RegTensor<float> v1, AscendC::Reg::RegTensor<float> v2, AscendC::Reg::MaskReg v3) {
// CHECK-NEXT:  AscendC::Reg::Abs(v1, v2, v3);
// CHECK-NEXT:  AscendC::Reg::Exp(v1, v2, v3);
// CHECK-NEXT:  AscendC::Reg::Ln(v1, v2, v3);
// CHECK-NEXT:  AscendC::Reg::Log(v1, v2, v3);
// CHECK-NEXT:  AscendC::Reg::Log10(v1, v2, v3);
// CHECK-NEXT:  AscendC::Reg::MaskNot(v1, v2, v3);
// CHECK-NEXT:  AscendC::Reg::Neg(v1, v2, v3);
// CHECK-NEXT:  AscendC::Reg::Not(v1, v2, v3);
// CHECK-NEXT:  AscendC::Reg::Relu(v1, v2, v3);
// CHECK-NEXT:  AscendC::Reg::Sqrt(v1, v2, v3);
// CHECK-NEXT:  AscendC::Reg::ReduceMax(v1, v2, v3);
// CHECK-NEXT:  AscendC::Reg::ReduceMin(v1, v2, v3);
// CHECK-NEXT:  AscendC::Reg::ReduceSum(v1, v2, v3);
// CHECK-NEXT:  AscendC::Reg::Duplicate(v1, v2, v3);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_unary_reg(%dst: !ascendc.reg_tensor<f32>, %src: !ascendc.reg_tensor<f32>, %mask: !ascendc.mask_reg) {
  ascendc.abs_reg %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.exp_reg %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.ln_reg %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.log_reg %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.log10_reg %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.mask_not_reg %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.neg_reg %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.not_reg %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.relu_reg %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.sqrt_reg %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.reduce_max_reg %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.reduce_min_reg %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.reduce_sum_reg %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.duplicate_reg %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  return
}

// CHECK-LABEL:void emit_vecscalar_reg(AscendC::Reg::RegTensor<float> v1, AscendC::Reg::RegTensor<float> v2, float v3, AscendC::Reg::MaskReg v4) {
// CHECK-NEXT:  AscendC::Reg::Adds(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::Muls(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::Maxs(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::Mins(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::LeakyRelu(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::ShiftLefts(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::Reg::ShiftRights(v1, v2, v3, v4);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_vecscalar_reg(%dst: !ascendc.reg_tensor<f32>, %src: !ascendc.reg_tensor<f32>, %scalar: f32, %mask: !ascendc.mask_reg) {
  ascendc.adds_reg %dst, %src, %scalar, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  ascendc.muls_reg %dst, %src, %scalar, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  ascendc.maxs_reg %dst, %src, %scalar, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  ascendc.mins_reg %dst, %src, %scalar, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  ascendc.leaky_relu_reg %dst, %src, %scalar, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  ascendc.shift_lefts_reg %dst, %src, %scalar, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  ascendc.shift_rights_reg %dst, %src, %scalar, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  return
}
