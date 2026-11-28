/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCVF2_OPS_TD
#define ASCVF2_OPS_TD

include "Dialect.td"
include "Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

include "asctile/Dialect/AscTile/IR/Attributes.td"

class AscVF2_Op<string mnemonic, list<Trait> traits = []> : Op<AscVF2_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::ascvf2";
}

def VFGroupOp : AscVF2_Op<
  "vf_group",
  [AffineScope, AttrSizedOperandSegments, SingleBlockImplicitTerminator<"::mlir::ascvf2::YieldOp">]
> {
  let summary = "Group vector-function operations for fused execution";
  let description = [{
    Groups a compatible use-def chain before lowering to AscendC. The operation
    exposes input, output, and temporary tensor segments and produces shaped
    values as results.
    AscVF2 uses UB local tensors before register lowering and AscendC local
    tensors after its boundary has been legalized. Its body is primarily SSA
    dataflow, but may also contain explicit synchronization operations such as
    ascvf2.barrier.
  }];
  let arguments = (ins Variadic<AnyShaped>:$inputTensors, Variadic<AnyShaped>:$outputTensors,
                   Variadic<AnyShaped>:$tempTensors);
  let results = (outs Variadic<AnyShaped>:$results);
  let regions = (region SizedRegion<1> : $region);
  let assemblyFormat = [{
    (`ins` `(` $inputTensors^ `:` qualified(type($inputTensors)) `)` )?
    (`outs` `(` $outputTensors^ `:` qualified(type($outputTensors)) `)` )?
    (`tmps` `(` $tempTensors^ `:` qualified(type($tempTensors)) `)` )?
    $region (`:` type($results)^)? attr-dict
  }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    bool canReuse(OpOperand&, OpOperand&);
  }];
}

def ToBufferOp : AscVF2_Op<"to_buffer", [Pure]> {
  let summary = "Expose a UB local tensor as a same-shaped UB memref";
  let description = [{
    Returns the backing UB memory of a static UB local tensor as a same-shaped
    memref. The operation gives AscVF2 an explicit local-tensor alias boundary
    while preserving the original tensor shape at the VF-group boundary.
  }];
  let arguments = (ins AnyType:$tensor);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($result)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def ToTensorOp : AscVF2_Op<"to_tensor", [Pure]> {
  let summary = "Materialize a UB local tensor from a same-shaped UB memref";
  let description = [{
    Materializes a UB local tensor handle from a same-shaped UB memref. It is
    the explicit tensor-side boundary for AscVF2's local-tensor alias semantics
    after affine and vector operations have written through the memref.
  }];
  let arguments = (ins AnyType:$buffer);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$buffer attr-dict `:` type($buffer) `to` type($result)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def BarrierOp : AscVF2_Op<"barrier"> {
  let summary = "Synchronize AscVF2 vector stores and loads";
  let description = [{
    Fences pending vector stores before subsequent vector loads. The operation
    is lowered to an AscendC local-memory barrier.
  }];
  let assemblyFormat = "attr-dict";
}

def ReduceOp : AscVF2_Op<"reduce", [Pure]> {
  let summary = "Reduce a vector to a one-element vector";
  let description = [{
    Defines the vector-to-vector reduction boundary used after lane-wise
    accumulation. The operation intentionally returns vector<1xT> rather than a
    scalar so later multi-dimensional reductions can preserve their result
    shape while staying in vector/register space.
  }];
  let arguments = (ins AnyType:$input, I32ArrayAttr:$dims, AscTile_ReduceKindAttr:$kind);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$kind $input $dims attr-dict `:` type($input) `to` type($result)";
  let hasVerifier = 1;
}

def SimpleMaskRegOp : AscVF2_Op<"simple_mask_reg", [Pure]> {
  let summary = "Ephemeral register mask before AscendC materialization";
  let description = [{
    Represents a one-dimensional register mask before choosing its final
    AscendC materialization. The element type is carried explicitly, and the
    mask count is either a static index attribute or a dynamic index operand;
    exactly one form is present.
  }];
  let arguments = (ins TypeAttr:$elementType, OptionalAttr<IndexAttr>:$count, Optional<Index>:$dynamicCount);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$elementType (`static` $count^)? (`dynamic` $dynamicCount^)? `:` type($result) attr-dict";
  let hasVerifier = 1;
}

class AscVF2_VecScalarOp<string mnemonic, string opSummary>
    : AscVF2_Op<mnemonic, [Pure, AllTypesMatch<["base", "result"]>]>
{
  let summary = opSummary;
  let arguments = (ins AnyType:$base, AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$base `,` $value attr-dict `:` type($base) `,` type($value)";
  let hasVerifier = 1;
}

class AscVF2_UnaryOp<string mnemonic, string opSummary>
    : AscVF2_Op<mnemonic, [Pure, AllTypesMatch<["operand", "result"]>]>
{
  let summary = opSummary;
  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
  let hasVerifier = 1;
}

def AddSOp : AscVF2_VecScalarOp<"adds", "Add a vector and a scalar">;
def SubSOp : AscVF2_VecScalarOp<"subs", "Subtract a scalar from a vector">;
def MulSOp : AscVF2_VecScalarOp<"muls", "Multiply a vector by a scalar">;
def DivSOp : AscVF2_VecScalarOp<"divs", "Divide a vector by a scalar">;
def MinSOp : AscVF2_VecScalarOp<"mins", "Compute the element-wise minimum with a scalar">;
def MaxSOp : AscVF2_VecScalarOp<"maxs", "Compute the element-wise maximum with a scalar">;
def ShLSOp : AscVF2_VecScalarOp<"shls", "Shift a vector left by a scalar">;
def ShRSOp : AscVF2_VecScalarOp<"shrs", "Shift a vector right by a scalar">;
def LeakyReluOp : AscVF2_VecScalarOp<"leaky_relu", "Compute leaky ReLU with a scalar slope">;
def ReluOp : AscVF2_UnaryOp<"relu", "Compute the element-wise ReLU of a vector">;
def NotOp : AscVF2_UnaryOp<"not", "Compute the bitwise NOT of a vector">;

def YieldOp : AscVF2_Op<"yield", [Terminator]> {
  let summary = "Yield results from an AscVF2 group";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
  let builders = [OpBuilder<(ins), "/* empty yield */">];
}

#endif // ASCVF2_OPS_TD
