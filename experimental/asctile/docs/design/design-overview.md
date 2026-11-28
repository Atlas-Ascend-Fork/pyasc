<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Design overview

> Published in March 2026, last updated in November 2026.

```{contents} Table of Contents
:local:
```

## 1. Goals & Scope

**AscTile** is a tile-based programming model for writing Ascend NPU kernels in Python. It sits at a higher abstraction level than the original PyAsc (`asc`) model, which mapped Python APIs 1:1 to Ascend C intrinsics.

**Core goals** (in priority order):
- Let developers express kernels in terms of *tensors* (ND-arrays in global memory) and *tiles* (fixed-shape chunks in on-chip memory). Buffer addresses, `TPipe`/`TQue` lifecycles, and synchronization barriers are not exposed to the user.
- Reach **≈90% of the performance of hand-optimized Ascend C operators** for representative workloads.
- Provide a NumPy-like operation set: arithmetic, reductions, shape manipulation, masking, atomics.
- **[Main engineering challenge]** Automate synchronization insertion, UB memory allocation, and ping-pong optimization through compiler passes and compiler hints.
- Support Ascend **A2/A3** (910B/910C platforms, C220 arch), and **A5** (950 platforms, C310 arch) hardware families.

**Relationship to PyAsc1 (`asc`):**
AscTile is implemented *on top of* the existing PyAsc infrastructure. `asctile.jit` uses a `JITFunction` subclass, a `FunctionVisitor` subclass, and a `Compiler` subclass; it reuses the base argument specialization, cache, launch, Ascend C emitter, and Bisheng compilation flow. The additions are the `asctile`, `ascvf`, and `ascvf2` MLIR dialects plus a dedicated pipeline that lowers tile-level IR to the `ascendc` dialect.

**Out of scope:**
- General-purpose, one-to-one access to the full Ascend C API. Limited interop with `asc` values and explicit code fragments is intentionally available as an escape hatch (`to_asc`, `from_asc`, `inline`, and `inline_vf`).
- Dynamic shapes at IR level — tile shapes must be statically known at JIT time; tensor shapes may be dynamic (runtime values).

---

## 2. Key Challenges for Ascend NPU

Ascend A2 AI Core has two types of compute cores:
- **AIC** (Cube) — matrix / tensor operations.
- **AIV** (Vector) — vector operations; 1 AIC : 2 AIV ratio on A2.

Each core has its own MTE (DMA), Scalar Unit, and Compute Unit. AIC and AIV have no direct interconnect — data exchange goes through L2 / Global Memory.

### 2.1 Synchronization Insertion

MTE and Compute Unit within each core run in parallel. Explicit `set_flag` / `wait_flag` barriers must be inserted between them. Missing barriers cause silent data hazards; redundant barriers cause stalls. Must be fully automated by the compiler.

### 2.2 Ping-Pong (Double Buffering)

MTE loads for tile N+1 must overlap with Compute on tile N. Requires the compiler to split the loop body, hoist loads, and insert correct barriers — without user annotation.

### 2.3 UB Memory Allocation and Reuse

On-chip UB (Unified Buffer) is ~256 KB per Da Vinci core (A2). After loop unrolling, many tile SSA values can be live simultaneously. The compiler must compute liveness and reuse freed UB regions. Tile shapes are statically known at JIT time; total live footprint must be validated at compile time.

### 2.4 Hardware Differences: A2/A3 vs A5

**A3** (910C) is two A2 (910B) dies in one package — same Da Vinci architecture, same AIC / AIV model, same challenges as A2. This section therefore focuses on A5.

A5 is not an incremental refresh of A2. It introduces new hardware capabilities (register-addressable SIMD, kernel-side printf) and two new Ascend C programming levels (covered in §9 and the companion *Programming Models Insights*). Findings below are derived from inspection of the CANN 9 SDK preview. <sup>[[58]](#ref-58)</sup>

#### 2.4.1 Hardware Capability Deltas

A5 adds the following hardware capabilities relative to A2: <sup>[[58]](#ref-58)</sup>

- **Register file is first-class.** The vector register file is now exposed as a planned address space for tile computation.
- **Low-precision matmul first-class.** Bit-mode matmul is now exposed as a dedicated hardware path, consistent with the new MX-family data formats (below).
- **New data formats.** MXFP4, MXFP8, HiF8 (in addition to FP16 / BF16 / INT8); the type-conversion matrix grew roughly 3× to cover the new dtypes, including saturating casts.
- **AIC↔AIV communication moves on-chip.** On A2/A3 there is no dedicated hardware channel between Cube and Vector cores — they exchange data through Global Memory (DMA in and out). On A5, a dedicated tag-based dual-channel FIFO lives in the consumer's on-chip SRAM to enable **zero-copy**. The PyPTO framework's TPUSH / TPOP protocol is the software layer built on top of this hardware. <sup>[[4]](#ref-4)</sup>
- **Cross-core address resolution.** On-chip ring-buffer placement means the producer must know the consumer's SRAM base. Resolved via per-function constants plus an allocator-reserved region in the consumer's SRAM.
- **Pipeline sync unchanged at the hardware level.** Handshake between memory engines and vector cores remains explicit — no new hardware barriers.

#### 2.4.2 Implications for the Three DSL Challenges

- **Sync insertion.** Three levels now coexist with different sync models: basic_api keeps explicit `set_flag` / `wait_flag` and TPipe events; Ascend C Reg (register API) adds `MaskReg` predication at the register-tile granularity (masks are not barriers — barriers remain basic_api's responsibility); SIMT-API adds warp-level primitives for fine-grained sync. A DSL must decide which model to expose (or hide) and stay coherent across levels.
- **Ping-pong.** The on-chip TPUSH / TPOP ring buffer on A5 removes the GM round-trip for Cube ↔ Vector handoff — the cost model of a pipelined stage shifts substantially. On register API, pipelining is a concern at register-tile granularity, not UB-tile granularity, because loops iterate over `RegTensor` chunks.
- **UB memory allocation.** Two new address-spaces to plan: the register file (first-class in register API) and the reserved consumer SRAM segment for the TPUSH / TPOP ring buffer (on A5, a fixed exclusion zone inside UB or L1). The DSL allocator must model both.
- **Portability.** Targeting only basic_api is the conservative choice but forfeits A5's register-file throughput. Targeting register API reaches the register file but requires c310 (A5-only builds). A DSL that claims to target A2 + A5 must lower to basic_api only, or implement per-target lowering.

---

## 3. Architecture Overview

```text
User Python code (@asctile.jit)
    │
    ▼
AscTile Python frontend
    experimental/asctile/python/asctile/language/
    experimental/asctile/python/asctile/runtime/
    │  GlobalTensor, LocalTensor, TensorLocation, range
    │  memory, arithmetic, reduction, shape, atomics, debug APIs
    │
    ▼
AST → MLIR
    FunctionVisitor (subclass of asc's Python AST visitor)
    pybind11 builders in experimental/asctile/python/src/OpBuilder.cpp
    │
    ▼
asctile MLIR dialect
    experimental/asctile/include/asctile/Dialect/AscTile/IR/
    experimental/asctile/lib/Dialect/AscTile/IR/
    │  tensor<..., #asctile.global>
    │  tensor<..., #asctile.local<LOCATION>>
    │  tile operations, matmul, reductions, regions, CV groups
    │
    ▼
Tile-level transforms and optional register fusion
    experimental/asctile/lib/Dialect/AscTile/Transforms/
    experimental/asctile/lib/Dialect/AscVF/Transforms/  (vf_fusion=1)
    experimental/asctile/lib/Dialect/AscVF2/Transforms/ (vf_fusion=2)
    │
    ▼
LowerToAsc conversion
    experimental/asctile/lib/Conversion/LowerToAsc/
    │  asctile + arith/math/scf/tensor → ascendc-compatible IR
    │
    ▼
ascendc MLIR dialect
    shared backend dialect plus extension passes in
    lib/Dialect/Asc/Transforms/ and
    experimental/asctile/lib/Dialect/AscendC/Transforms/
    │  allocation, reuse, CV placement, synchronization, boilerplate
    │
    ▼
Ascend C source
    lib/Target/AscendC/CodeEmitter.cpp
    │
    ▼
Bisheng compiler
    │
    ▼
NPU kernel object and runtime launch
```

The extension is built in `experimental/asctile/`, but its Python package is installed and imported as `asc.experimental.asctile`. It reuses the root PyAsc AST visitor, runtime, target, and Bisheng integration rather than maintaining a separate host runtime.

### End-to-end compilation flow

```
@asctile.jit kernel invocation
  → argument type specialization  (same as asc)
  → two-level cache lookup         (same as asc)
      → [miss]
          AST walk → asctile MLIR (FunctionVisitor + pybind11 builders)
          ↓
          Tile-level transforms (location, CV strategy, matmul, FP8, math)
          optional AscVF2 register fusion (vf_fusion=2)
          ↓
          LowerToAsc passes (asctile → ascendc-compatible IR)
          ↓
          optional AscVF fusion (vf_fusion=1)
          ↓
          ascendc passes (allocation, reuse, sync, boilerplate)
          ↓
          CodeEmitter     (MLIR → Ascend C source)
          ↓
          Bisheng         (Ascend C → .o binary)
          ↓
          cache store
  → kernel launch via runtime rt library calls  (same as asc)
```

When the `ASCTILE_INTERPRET` environment variable is set to `1`, `on`, or `true`, `asctile.jit` returns an interpreted function instead of invoking this compilation path. The interpreter is a debugging aid and does not represent the performance model.

---

## 4. Programming Model

### 4.1 Core tensor types

| | `GlobalTensor` | `LocalTensor` |
|--|----------|--------|
| **Memory** | Global memory (HBM/GM) | On-chip local memory selected by `TensorLocation` |
| **Shape** | ND; dimensions may be runtime values (`RuntimeInt`) | ND; static at JIT time |
| **Creation** | `asctile.global_tensor(base, shape, l2_cache=True)` | Returned by loads, copies, creation ops, and most compute ops |
| **Semantics** | A descriptor for a contiguous external buffer | Value semantics: operations produce new SSA tiles |
| **IR form** | `tensor<shape x dtype, #asctile.global>` | `tensor<shape x dtype, #asctile.local<LOCATION>>` |

`GlobalTensor` carries the base address, shape, dtype, and L2-cache policy. `LocalTensor` is safe to CSE, copy, and move; physical allocation is introduced later by the compiler. `tile.to(dtype, round_mode=...)` converts between dtypes, while `tile.to(location)` expresses a location conversion.

### 4.2 Memory hierarchy (`TensorLocation`)

| Value | Typical use |
|------------|-------------|
| `Auto` | Leave the destination unresolved; `ResolveAutoLocation` selects a concrete location (usually `UB`) |
| `UB` | Unified Buffer; vector computation and scalar-tile operations |
| `L1` | Local staging, including the GM → L1 → L0 path for cube operands |
| `L0A` / `L0B` | Cube matmul input buffers |
| `L0C` | Cube matmul accumulator/output buffer |
| `FIX` | Fixpipe-related local storage |
| `BT` | Bias table for matmul bias initialization |

`copy_in` accepts `TensorLocation.Auto`, `UB`, `L1`, `L0A`, `L0B`, and `BT`. Loads directly to `L0A`/`L0B`/`BT` are split into legal GM → L1 → destination paths. `copy` supports explicit local-to-local transfers such as L1 → L0A/L0B/BT, L0C → L1/UB, and UB → L1, including vector-sub-block distribution modes.

### 4.3 Load, store, and local copies

```python
# Tensor load; offsets are element offsets, and shape must be static.
tile = asctile.copy_in(tensor, [base], [128])

# Scalar load: omit shape.
scalar = asctile.copy_in(tensor, [i])

# Padding. real_shape says how much of the tile is live; the rest is pad_value.
tile = asctile.copy_in(tensor, [base], [128], real_shape=[97], pad_value=0.0)

# Local-to-local copy, subregion, relocation, and optional distribution mode.
clone = asctile.copy(tile)
sub = asctile.copy(tile, [0], [64])
l0a = asctile.copy(l1_tile, [0, 0], [16, 32], TensorLocation.L0A)

# Store a tile, scalar, or one-element tile. real_shape supports partial stores.
asctile.copy_out(tile, tensor, [base], real_shape=[97])
asctile.copy_out(scalar, tensor, [i])
```

For 2D and higher tiles in `UB`, the last dimension must be aligned to 32 bytes. `copy_in` and `copy_out` also support `gather` and `scatter` along a non-final axis; `reindex` gathers individual elements from a local tensor. Higher-rank load/store support remains experimental.

### 4.4 Public API overview

`asc.experimental.asctile` exports the following API families; the generated Python reference is the complete, authoritative list.

| Family | Public names |
|--------|--------------|
| Tensor and memory | `GlobalTensor`, `LocalTensor`, `TensorLocation`, `RoundMode`, `DistribMode`, `global_tensor`, `copy_in`, `copy_out`, `copy`, `gather`, `scatter`, `reindex` |
| Arithmetic and comparison | `add`, `sub`, `mul`, `div`, `pow`, `maximum`, `minimum`, `equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `left_shift`, `right_shift`; corresponding `LocalTensor` operators |
| Unary math and normalization | `abs`, `ceil`, `floor`, `negative`, `relu`, `sqrt`, `rsqrt`, `exp`, `exp2`, `log`, `log2`, trigonometric/hyperbolic functions, `erf`, `softmax`, `rms_norm`, `layer_norm`, `bitwise_not` |
| Reductions | `reduce_sum`, `reduce_max`, `reduce_min`, `reduce_prod`; `sum`, `max`, `min`, and `prod` methods with optional axes and `keep_dims` |
| Shape | `broadcast_shapes`, `broadcast_tensors`, `broadcast_to`, `reshape`, `ravel`, `expand_dims`, `squeeze`, `transpose` |
| Creation and casts | `full`, `full_like`, `zeros`, `zeros_like`, `zeros_acc`, `concat`, `cast`, `LocalTensor.to` |
| Matrix multiplication | `matmul`, `matmul_acc`; `zeros_acc` creates the L0C accumulator and can initialize it with a `BT` bias |
| Masking and selection | `mask`, `where` |
| Atomics | `atomic_add`, `atomic_max`, `atomic_min` |
| Programming model | `block_idx`, `block_num`, `sub_block_idx`, `sub_block_num`, `cv_strategy` |
| Control flow | `range`, `static_range` |
| Low-level interop | `empty`, `empty_like`, `from_asc`, `to_asc`, `inline`, `inline_vf` |
| Debug | `device_assert`, `device_print`, `static_assert`, `static_print` |
| Runtime and launch | `jit`, `autotune`, `CompileOptions`, `set_platform`, `Backend`, `Platform`, `KernelType`, `ConstExpr`, `GlobalAddress`, dtype constants |

Dtype promotion follows the existing `infer_common_dtype` behavior. Most operations produce a new value; explicit low-level interop and atomic/store operations are the intentional mutation/escape-hatch exceptions.

### 4.5 `asctile.range`

`asctile.range` supports Python-style one-, two-, and three-argument forms and two compiler controls:

```python
for i in asctile.range(start, stop, step, unroll_factor=4, gm_barrier=False):
    ...
```

- `unroll_factor` — how many iterations to unroll. Tagged on the MLIR `scf.for` operation and processed by `UnrollLoop`.
- `gm_barrier=True` — prevents overlapping GM accesses across iterations when iteration N reads data written by iteration N-1.

`asctile.static_range` requires compile-time bounds and enables complete static unrolling. Plain Python `range` is remapped to `asctile.range` inside a JIT function.

---

## 5. IR Design

### 5.1 `asctile` dialect

Defined in `experimental/asctile/include/asctile/Dialect/AscTile/IR/` (TableGen declarations) and implemented in `experimental/asctile/lib/Dialect/AscTile/IR/`.

**Types:**

| IR type | Python proxy |
|---------|--------------|
| `tensor<[M,N] x f32, #asctile.global>` | `GlobalTensor` |
| `tensor<[128] x f16, #asctile.local<UB>>` | `LocalTensor` |

Both are MLIR ranked tensor types with an `asctile` encoding. Global dimensions may be dynamic (`?`); local dimensions are static. `#asctile.local<LOCATION>` carries the `TensorLocation` enum (`Auto`, `L0A`, `L0B`, `L0C`, `UB`, `L1`, `FIX`, or `BT`). A logical location is not yet a physical address.

**Key operations (from `Ops.td`):**

| Category | Op | Description |
|----------|----|-------------|
| Tensor creation | `tensor` | Create a `GlobalTensor` from a base pointer, runtime sizes, and L2-cache policy |
| | `dim` | Read a dynamic dimension from a `GlobalTensor` at runtime |
| | `accumulator` | Create an L0C accumulator; optionally initialize it from a `BT` bias tile |
| Memory transfer | `load` | Load a static-shape tile or padded tile from a `GlobalTensor` |
| | `store` | Store a `LocalTensor` to a `GlobalTensor`, optionally with `real_shape` |
| | `store_fixpipe` | Store an L0C tile through the fixpipe path |
| | `copy` | Copy a tile/subregion to a new tile, optionally relocating and distributing it |
| | `copy_fixpipe` | Copy an L0C tile through fixpipe |
| | `get_value` / `set_value` | Scalar load/store from/to a `GlobalTensor` |
| | `gather` / `scatter` | Gather/scatter subtensors between `GlobalTensor` and `LocalTensor` |
| | `reindex` | Gather individual elements from a local tensor by index |
| Element-wise compute | `cast` | Type conversion between tile dtypes |
| | `relu` | Fused ReLU |
| | `adds`, `subs`, `muls`, `divs`, `mins`, `maxs`, `shls`, `shrs`, `leaky_relu`, `cmps` | Vector-scalar compute |
| | `power`, `bitwise_not` | Tile-wise power and bitwise complement |
| Reduction / norm | `reduce` | Reduce tile along one or more axes |
| | `reduce_as_1d` | Reduce a tile to a 1-element tile or a scalar |
| | `softmax` | Row-wise softmax on a 1D/2D tile |
| | `rms_norm` / `layer_norm` | Normalisation operations |
| Matmul | `matmul` | 2-D matrix multiply |
| | `matmul_acc` | 2-D matrix multiply accumulating into an L0C tile |
| Shape | `reshape` | Reshape a tile |
| | `broadcast` | Broadcast tile to a wider shape |
| | `transpose` | Permute tile dimensions |
| Atomic | `atomic_rmw` | Atomic read-modify-write (add / max / min) |
| Region / control | `count_mask`, `bitwise_mask` | Regions guarded by element count or bitmask |
| | `cube_group` / `vector_group` | Group operations for Cube and Vector sides of a CV kernel |
| | `cv_strategy` | Express a split/join strategy between Cube and Vector sub-blocks |
| | `yield` | Region terminator |
| Interop / debug | `inline`, `inline_vf` | Emit explicit Ascend C or VF code fragments |
| | `assert`, `dump_tensor` | Device assertions and dump hooks |

Scalar arithmetic and control flow use standard MLIR `arith`, `math`, and `scf` operations. Tile-tensor arithmetic initially also uses standard shaped-tensor operations where possible; `TransformMathOps` specializes them into `asctile` vector-scalar operations before lowering.

### 5.2 IR generation

The base AST visitor is `python/asc/codegen/function_visitor.py`. AscTile specializes it in `experimental/asctile/python/asctile/runtime/codegen.py`; builder methods exposed through `experimental/asctile/python/src/OpBuilder.cpp` create `asctile.*` operations. Public API calls generally lower to one op or a small fixed sequence. No memory allocation or synchronization decisions are made during AST construction.

### 5.3 Lowering chain

```text
asctile dialect + standard MLIR
    ↓ AscTile transforms (location, CV, cube, math, FP8, masking)
    ↓ optional AscVF2 transforms (vf_fusion=2)
    ↓ LowerToAsc transforms (asctile → ascendc-compatible operations)
    ↓ optional AscVF transforms (vf_fusion=1)
ascendc dialect + emitasc/scf/arith/math
    ↓ AscendC transforms (allocation, reuse, sync, boilerplate)
Ascend C source
```

---

## 6. Optimization Passes

The pipeline is scheduled by `Compiler.schedule_passes()` in `experimental/asctile/python/asctile/runtime/compiler.py`. Paths below are relative to the repository root. The generated pass references under `experimental/asctile/docs/mlir/passes/` contain the detailed transformation rules.

### 6.1 AscTile transforms (`experimental/asctile/lib/Dialect/AscTile/Transforms/`)

These operate while tile-level types and operations are still present.

| Pass | Purpose |
|------|---------|
| `ResolveAutoLocation` | Select concrete locations for `TensorLocation.Auto` operands/results |
| `LocationCastToCopy`, `FulfillDataTransfer` | Materialize legal location conversions and data transfers |
| `VerifyTensorLocation` | Reject unsupported locations and transfer combinations |
| `SplitCubeLoad` | Split direct GM → L0A/L0B/BT loads into the legal staged path |
| `PrepareCVStrategy`, `ApplyCVStrategy`, `WrapCVGroups`, `MergeCVGroups` | Partition and annotate Cube/Vector work for mixed-core kernels |
| `CombineExtractSlice` | Fold eligible `tensor.extract_slice` operations into producers |
| `CubeTransposeToLoad`, `VectorTransposeToLoadStore` (C310) | Fold transposes into load/store/copy attributes |
| `LegalizeMatmul`, `MarkMatmulAccWithBias`, `DetectBiasLoad` | Legalize matmul shapes, bias, and BT data flow |
| `ApplyHomomorphism` | Move suitable scalar operations across supported tile operations |
| `FoldCast`, `ReplaceFp8Ops` | Simplify casts and legalize FP8 operation forms |
| `TransformMathOps` | Specialize standard arithmetic/math operations into tile-aware forms |
| `TransformStoreFixpipe` | Convert eligible L0C stores into fixpipe stores |
| `MarkReuseSource` | Annotate values that can participate in memory reuse |
| `UnscalarizeReduction` (C310) | Replace scalar-tail reductions with vector forms |
| `UnrollLoop` | Unroll `scf.for` loops tagged with `unroll_factor` |

MLIR `Canonicalizer`, `CSE`, and `ReconcileUnrealizedCasts` are interleaved between these transformations.

### 6.2 Vector-function fusion

VF fusion is available only on C310.

| Mode | Passes | Behavior |
|------|--------|----------|
| `vf_fusion=0` | None | Keep ordinary UB-based operations |
| `vf_fusion=1` | `PromoteCastOps`, `FindVFGroup`, `CanonicalizeVFGroup`, `LowerToReg`, `DispatchVFFusion`, `MaterializeLoadStore`, `InlineVFGroup` from `experimental/asctile/lib/Dialect/AscVF/Transforms/` | Legacy register-level VF-block path |
| `vf_fusion=2` | `OutlineVFGroup`, `ShrinkVFSignature`, `Vectorize`, `ReorderGroupedOps`, `FuseVFLoops`, `PromoteMemToReg`, `DropDeadWrite`, `MergeIntoTransfer`, `InsertVFSync`, `LowerToReg`, and related passes from `experimental/asctile/lib/Dialect/AscVF2/Transforms/` | Modern path; runs before LowerToAsc and includes affine loop normalization/fusion |

### 6.3 LowerToAsc passes (`experimental/asctile/lib/Conversion/LowerToAsc/`)

These lower `asctile.*` operations into `ascendc.*` plus standard MLIR operations.

| Pass | Purpose |
|------|---------|
| `ExpandMath` | Replace `math.*` operations with `ascendc` vector intrinsic sequences |
| `RedressI1Tensor` | Widen boolean (i1) tensors to `i8`/`ui8` as required by Ascend C |
| `LowerArith` | Lower `arith.*` scalar ops to `ascendc` equivalents |
| `LowerArithBinary` | Lower tile-scalar and scalar-tile arithmetic |
| `LowerAtomic` | Lower `AtomicRMWOp` → `ascendc.atomic_*` |
| `LowerAscTileDataTransfer` | Lower load/store/copy/fixpipe variants to `ascendc` data-copy operations |
| `LowerAscTile` | Main tile-op lowering: casts, reductions, normalizations, and related operations |
| `LowerAscTileToBasic` | Lower high-level tile operations to basic ascendc forms |
| `LowerMath` | Lower remaining `math.*` ops |
| `LowerSCF` | Lower `scf.for` loops to `ascendc`-compatible form |
| `LowerTensor` | Lower remaining tensor operations |
| `DisplaceConcat` | Replace `tensor.concat` with adjacent allocations / copies |
| `RealizeConversionCast` | Materialise any pending unrealized casts from the lowering |
| `ExpandMask` | Lower `count_mask`/`bitwise_mask` regions into mask sequences |

### 6.4 AscendC passes

Once `asctile` operations are gone, the backend pipeline takes over. Extension-specific passes live in `experimental/asctile/lib/Dialect/AscendC/Transforms/`; shared base passes live in `lib/Dialect/Asc/Transforms/`.

| Phase | Key passes |
|-------|-----------|
| **Initial cleanup** | `PrivatizeFunc`, `Inliner`, `SymbolDCE`, `Canonicalizer` |
| **Lowering glue** | `PromoteCVBlock`, `FillAscOperands`, `FixupMmadAccParams`, `InputOutputTensor`, `RefineCubePosition` |
| **Memory** | `ReuseUBAllocation` (`reuse_alloc=1`), `ComputeReuseGroup`, `HoistTensorAllocation`, `ReuseTensorAllocation` (`reuse_alloc=2`), `DispatchAlloc`, `UnifyBiasTensor`, `UnifyPipe` |
| **Optimization** | `UnrollLoop`, `Canonicalizer`, `CSE`, `LICM`, `SCCP` |
| **Cross-core and pipe sync** | `InsertCrossCoreSync`, `InsertCrossCoreSyncGM`, `EraseSync`, `HoistQueBind`; `InsertQueSync` on C220, or `InsertBufIdSync`, `InsertBiasBufIdSync`, and `FuseBufIdSync` on C310 |
| **Postprocessing** | `InlineVFGroup`, `WrapVecScope`, `DeclarePyStruct`, `GenerateBoilerplate`, `InsertSubBlockGuard` (`cv_ratio=1`), `DefineCubeOnly` (`matmul_cube_only=True`), `LegalizeKernelArgs`, `DetectKernelType`, `InsertInitDump`, `VerifySync` (`verify_sync=True`), `StripDebugInfo` (`strip_loc=True`), `ComputeMemoryConsumption` |

`ComputeMemoryConsumption` is scheduled by the AscTile compiler and records per-`TPosition` usage as a module attribute; it raises a compile-time error on overflow.

---

## 7. Memory & Resource Model

### 7.1 LocalTensor-level view

From the user's perspective, memory management is invisible. The user:
1. Creates `GlobalTensor` descriptors pointing to HBM buffers.
2. Calls `copy_in(tensor, offsets, shape, location=…)` to bring data into a `LocalTensor`.
3. Applies operations to produce new tiles.
4. Calls `copy_out(tile, tensor, offsets, real_shape=…)` to write results back.

The compiler is responsible for mapping each `LocalTensor` SSA value to a physical UB region.

### 7.2 Compiler-side UB allocation

Two strategies, selected per compilation:

| Strategy | Option | How it works |
|----------|--------|-------------|
| **TPipe-backed** | `static_alloc=False` | `MaterializeTensor` emits TPipe-backed allocation in generated Ascend C. It is more flexible but adds scalar overhead. Default on C220. |
| **Static** | `static_alloc=True` | `AllocateTensor` computes fixed offsets at compile time. It requires static tile shapes. Default on C310, except for kernels containing an incompatible `BroadcastOp`, for which `DispatchAlloc` falls back to TPipe. |

`static_alloc` is `Optional[bool]`. When it is `None`, the module attribute is omitted and `DispatchAlloc` selects the strategy from the target architecture and operation mix. Explicit `False` forces TPipe-backed allocation; explicit `True` forces static allocation.

**UB pressure reduction (both modes):**
- `HoistTensorAllocation` — move allocations above loops so one allocation covers all iterations.
- `ReuseUBAllocation` (`reuse_alloc=1`) — legacy liveness-based reuse, applied both to input/output tensors and ordinary tiles.
- `ReuseTensorAllocation` (`reuse_alloc=2`) — modern reuse based on reuse groups; recommended for most kernels.
- `ComputeMemoryConsumption` — sums all live tile sizes per `TPosition` and fails the compilation if UB limits are exceeded.

### 7.3 Synchronization

Because asctile users don't write `set_flag`/`wait_flag`, all synchronization is inserted by the compiler:

- `insert_sync=True` is set automatically by `asctile.jit`.
- On **C220**: `InsertQueSync` uses queue-based synchronization.
- On **C310**: `InsertBufIdSync` tracks buffer IDs instead of queue positions; `InsertBiasBufIdSync` handles BT bias tensors, and `FuseBufIdSync` merges adjacent synchronization operations.
- `verify_sync=True` runs an additional `VerifySync` pass after sync insertion as a sanity check.
- `auto_sync=True` independently enables Bisheng's `--cce-auto-sync`; AscTile's MLIR `insert_sync` pass is not a replacement for that compiler option.

---

## 8. Usage Examples

### Vector addition

```python
from asc.experimental import asctile

@asctile.jit
def vector_add(x_ptr: asctile.GlobalAddress,
               y_ptr: asctile.GlobalAddress,
               out_ptr: asctile.GlobalAddress,
               size: int,
               tile_size: asctile.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])

    tiles_per_core = asctile.ceildiv(asctile.ceildiv(size, asctile.block_num()), tile_size)
    core_offset = asctile.block_idx() * tiles_per_core * tile_size

    for i in asctile.range(tiles_per_core, unroll_factor=2):
        offset = core_offset + i * tile_size
        x = asctile.copy_in(x_gm, [offset], [tile_size])
        y = asctile.copy_in(y_gm, [offset], [tile_size])
        asctile.copy_out(x + y, out_gm, [offset])

vector_add[8](x, y, out, x.numel(), 128)
```

### Softmax (row-wise)

```python
@asctile.jit(reuse_alloc=2, vf_fusion=2)
def row_wise_softmax(rows: asctile.LocalTensor) -> asctile.LocalTensor:
    row_max = asctile.reduce_max(rows, 1, keep_dims=True)
    shifted = rows - row_max
    exp_vals = asctile.exp(shifted)
    return exp_vals / asctile.reduce_sum(exp_vals, 1, keep_dims=True)

@asctile.jit(reuse_alloc=2, vf_fusion=2)
def fused_softmax(input_ptr, output_ptr, num_rows, num_cols, tile_shape: asctile.ConstExpr):
    in_gm = asctile.global_tensor(input_ptr, [num_rows, num_cols])
    out_gm = asctile.global_tensor(output_ptr, [num_rows, num_cols])

    rows_per_block = asctile.ceildiv(num_rows, asctile.block_num())
    block_offset = asctile.block_idx() * rows_per_block
    ub_loop = asctile.ceildiv(rows_per_block, tile_shape[0])

    for i in asctile.range(ub_loop, unroll_factor=2):
        row_start = block_offset + i * tile_shape[0]
        rows = asctile.copy_in(
            in_gm,
            [row_start, 0],
            [tile_shape[0], tile_shape[1]],
            pad_value=float("-inf"),
        )
        asctile.copy_out(row_wise_softmax(rows), out_gm, [row_start, 0])
```

The softmax tile width must be 32-byte aligned. For a runtime column count that is not aligned, round the tile width up and pad with `-inf`, the identity for `max`.
`vf_fusion=2` in this example requires a C310 (`Ascend950PR_95*`) target.

---

## 9. Insights from Existing Programming Models

A comparative analysis of existing tile-based and kernel DSLs relevant to Ascend NPU (AscendC, Triton, cuTile, TileLang-Ascend, Triton-Ascend, PyPTO, Pallas, Mojo) is maintained in a companion document:

→ [Programming Models Insights](programming-models-insights.md)

The observations from that document inform the design decisions in §10.

---

## 10. Key Design Decisions

- **Logical tiles before physical buffers.** The frontend models `LocalTensor` as a value with a logical location. Allocation, reuse, and synchronization remain compiler decisions until after tile-level transformations and unrolling.
- **A dedicated tile dialect.** Keeping tile operations distinct from `ascendc` exposes location, CV grouping, matmul, and fusion intent to transformations that would be difficult or impossible after scalar/API lowering.
- **Architecture-specific backends, one frontend.** C220 and C310 share the Python model and most tile-level IR, while synchronization, VF, register, and allocation strategies diverge in later passes.
- **Optimization is selectable.** Memory reuse and VF fusion have multiple implementations because aggressive transformations can trade memory pressure, code size, and performance differently by kernel.
- **Escape hatches are explicit.** `to_asc`, `from_asc`, `inline`, and `inline_vf` provide interop without making the whole Ascend C API part of the high-level contract.
- **Debugging does not alter the programming model.** Interpreter mode, device debug operations, pass IR dumping, and sync verification are controlled separately from kernel semantics.

---

## 11. References

<a id="ref-4"></a>**[4]** PTO ISA — TPUSH/TPOP protocol.
<https://github.com/hw-native-sys/pypto/blob/main/docs/en/reference/pto-isa/01-tpush_tpop.md>

<a id="ref-58"></a>**[58]** CANN 9 SDK preview — 950 / A5 programming model findings (internal notes from SDK source inspection).

---

## Appendix: `CompileOptions` for AscTile

`asc.experimental.asctile.CompileOptions` extends `asc.runtime.compiler.CompileOptions`. All fields can be passed to `asctile.jit(**options)` or to `asctile.autotune` configurations.

| Option | Type/default | Effect |
|--------|--------------|--------|
| `cv_ratio` | `Literal[1, 2]`, default `1` | Number of vector sub-cores per cube sub-core; drives CV grouping and `InsertSubBlockGuard` |
| `debug` | `bool`, default `False` | Keep `device_print`/`device_assert`; when false, `RemoveDebugOps` deletes them |
| `insert_sync` | `bool`, default `True` | Run AscTile's MLIR synchronization-insertion pipeline |
| `reuse_alloc` | `Literal[0, 1, 2]`, default `0` | `0`: disabled; `1`: legacy `ReuseUBAllocation`; `2`: modern `ReuseTensorAllocation` (recommended) |
| `static_alloc` | `Optional[bool]`, default `None` | `True`: force static allocation; `False`: force TPipe-backed allocation; `None`: let `DispatchAlloc` choose |
| `vf_fusion` | `Literal[0, 1, 2]`, default `0` | `0`: disabled; `1`: legacy AscVF; `2`: modern AscVF2. C310 only |

Other options are inherited from the existing `CompileOptions`.
