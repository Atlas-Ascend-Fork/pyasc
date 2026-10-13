/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCVF_OPS_TD
#define ASCVF_OPS_TD

include "Dialect.td"
include "Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class AscVF_Op<string mnemonic, list<Trait> traits = []> : Op<AscVF_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::ascvf";
}

def LoadOp : AscVF_Op<"load"> {
  let description = "Load RegTensor from LocalTensor";
  let arguments = (ins AnyTypeOf<[AscendC_MaskReg, AscendC_RegTensor]>:$dstReg, AscendC_LocalTensor:$srcTensor,
                   Index:$offset, AscendC_MaskReg:$mask);
  let assemblyFormat = [{
    $dstReg `,` $srcTensor `[` $offset `]` `,` $mask attr-dict `:` type($dstReg) `,`
    type($srcTensor) `,` type($offset) `,` type($mask)
  }];
  let hasVerifier = 1;
}

def StoreOp : AscVF_Op<"store"> {
  let description = "Store RegTensor to LocalTensor";
  let arguments = (ins AscendC_LocalTensor:$dstTensor, Index:$offset,
                   AnyTypeOf<[AscendC_MaskReg, AscendC_RegTensor]>:$srcReg, AscendC_MaskReg:$mask);
  let assemblyFormat = [{
    $dstTensor `[` $offset `]` `,` $srcReg `,` $mask attr-dict `:` type($dstTensor) `,`
    type($offset) `,` type($srcReg) `,` type($mask)
  }];
  let hasVerifier = 1;
}

def BarrierOp : AscVF_Op<"barrier"> {
  let description = "Memory barrier";
  let assemblyFormat = "attr-dict";
}

def VFGroupOp : AscVF_Op<"vf_group", [SingleBlockImplicitTerminator<"::mlir::ascvf::YieldOp">, AttrSizedOperandSegments
]> {
  let summary = "block with fused operations";
  let arguments = (ins Variadic<AscendC_LocalTensor>:$dstList,
                   Variadic<AscendC_LocalTensor>:$srcList, TypeAttr:$groupType);
  let regions = (region SizedRegion<1> : $region);
  let assemblyFormat = [{
    (`dst` `(` $dstList^ `:` qualified(type($dstList)) `)` )? (`src` `(` $srcList^
    `:` qualified(type($srcList)) `)`)? $region attr-dict
  }];
  let hasVerifier = 1;
}

def YieldOp : AscVF_Op<"yield", [Terminator]> {
  let assemblyFormat = "attr-dict";
}

#endif // ASCVF_OPS_TD
