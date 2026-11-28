---
name: asctile-develop-vf-fusion
description: Use when developing or debugging AscVF2, including vf_fusion=2 passes, vectorization, fusion, and lowering.
---

# AscTile: Develop VF Fusion (AscVF2)

## When to Use

Use this skill for AscVF2 (`vf_fusion=2`) development, debugging, refactoring, and testing. For initial environment
setup, use `pyasc-check-env-and-build`; for a new pass skeleton, use `asctile-create-empty-pass`.

## Core Model

AscVF2 is the `vf_fusion=2` replacement for legacy AscVF. It lowers vector functions through standard MLIR vector,
affine, SCF, and memref IR before emitting AscendC register operations.

```text
outline pure VF groups
  -> shrink call signatures
  -> vectorize local tensors
  -> affine cleanup/fusion/coalescing
  -> exact FuseVFLoops
  -> store-to-load promotion and dead-write cleanup
  -> broadcast-transfer merging
  -> synchronization
  -> AscendC register lowering
  -> register-tensor reuse and ephemeral mask expansion
```

Pipeline order is defined in `experimental/asctile/python/asctile/runtime/compiler.py`. Passes are under
`experimental/asctile/lib/Dialect/AscVF2/Transforms/`; FileCheck tests are under
`experimental/asctile/test/Dialect/AscVF2/Transforms/`.

## Workflow

1. Locate the pass and its focused FileCheck test using the source map.
2. Update implementation and test together for semantic changes.
3. Build `asctile-opt`, then run the focused test followed by the full AscVF2 lit suite.
4. For Python-facing changes, rebuild the Python extension after building `asctile-opt`.

## Pipeline Invariants

- Generated VF functions are gated by `ascvf2.vector_func`; outlining and caller-side group legalization are not.
- `OutlineVFGroup` initially creates `ascvf2.vf_group` regions and vector function calls with results. Later passes
  make groups result-free; `ShrinkVFSignature` removes unused call results while keeping arguments and the symbol.
- `vf_group` tensor operand lists are metadata for later interface lowering: inputs, outputs, and temporaries. They
  do not carry loop-carried results.
- `LegalizeVFGroup` hoists only zero-operand `ascendc.localtensorauto` operations above the group, preserving their
  relative order.
- `AssignVFGroupTensors` classifies captured local tensors by read/write behavior and post-group liveness: live
  read-write tensors are both inputs and outputs, while dead-after-group tensors are temporaries.
- Intermediate storage is UB `memref.alloca`, not `tensor.empty` plus buffer conversion.
- Do not fold arbitrary `ascvf2.to_buffer` / `ascvf2.to_tensor` alias boundaries or equate distinct allocations;
  exact inverse pairs are intentionally folded.
- `ReorderGroupedOps` moves pure operations to the earliest valid position, sinks `to_tensor`, hoists allocations,
  and hoists zero-operand reinterpret casts with empty offsets, sizes, and strides without changing dataflow.

## Vectorization

- Hardware width is `vector_bytes / element_size`; vector bytes default to `256` and may be set by
  `asc.vf_vec_len`.
- Ordinary elementwise local tensors flatten to one-dimensional maximum-width vectors to reduce loop iterations.
- A trailing-axis broadcast with multiple distinct source values preserves row shape so producer and row-aware
  consumer can fuse.
- Reductions accumulate in maximum-width vectors, then `ascvf2.reduce` to `vector<1>`.
- Partial tails use `vector.create_mask` or `vector.constant_mask`. Full masks use a `CreateMask` enum pattern when
  representable and `UpdateMask` otherwise.

## Memory Cleanup

- `PromoteMemToReg` forwards an exact unmasked store to a later matching load. It also forwards a `vector<1>` store
  to a constant-zero broadcast load by emitting `vector.broadcast`.
- Promotion checks tensor alias root, indices, affine maps, element type, bounds, masks, and intervening side effects.
  A write invalidates pending writes with the same alias root. Unsupported cases remain memory operations.
- `DropDeadWrite` erases a write whose private tensor root has no later observable read, after checking the
  alias-user chain.
- `MergeIntoTransfer` combines a `vector<1>` read plus broadcast into a broadcast `vector.transfer_read` with a
  constant-zero permutation map.

## Loop Fusion

`FuseVFLoops` must answer dependence directly, not pattern-match affine syntax.

- Candidate loops are consecutive, in the same block, with identical affine lower/upper maps, operands, and step.
- Bodies may contain loops, yields, masks, transfers, and pure operations.
- For every constant iteration, compose and evaluate all transfer indices.
- Resolve each transferred memref through `to_buffer` / `to_tensor` / reinterpret chains to its originating
  allocation. Build exact offset sets per allocation, including only active mask lanes.
- Reject fusion if one iteration reads or writes data written by a different iteration. Same-iteration
  producer/consumer overlap is allowed and handled later by store-to-load promotion.
- Unknown bounds, layouts, and non-affine values reject fusion. Unevaluable masks are treated as full-width accesses;
  unevaluable mask operands reject fusion.

The pipeline runs VF fusion before affine loop coalescing, then coalesces and runs VF fusion again. Same-domain outer
loops merge first; flattening then exposes additional compatible loops.

Do not merge loops merely because their full accessed-address unions match. If producer and consumer group those
addresses differently per iteration, fusion can interleave cross-iteration reads and writes.

## Register Lowering

- `LowerToReg` converts vector operations to `ascendc.reg_tensor` operations.
- Contiguous transfers use normal load/store distributions.
- Broadcast reads use `DIST_BRC_B8`, `DIST_BRC_B16`, or `DIST_BRC_B32` by element size.
- A contiguous `vector<1>` store uses `DIST_FIRST_ELEMENT_B8`, `DIST_FIRST_ELEMENT_B16`, or
  `DIST_FIRST_ELEMENT_B32` by element size.
- Other stores use `DIST_NORM`.
- A null `vector::MaskOp` means an unmasked source op; conversion must create a full mask.
- Reduction identities must be type-correct: sum uses zero, max uses negative infinity, and min uses positive infinity.
- `ReuseRegTensor` combines same-type non-overlapping loop-aware register intervals only when the surviving allocation
  dominates all rewritten uses; values read before their first write in a loop remain live for that loop. Branching
  functions and region-crossing or terminator uses are rejected.

## Source Map

| Concern | Location |
| --- | --- |
| Pipeline order | `experimental/asctile/python/asctile/runtime/compiler.py` |
| Vectorization | `Transforms/Vectorize.cpp` |
| Loop fusion | `Transforms/FuseVFLoops.cpp` |
| Memory promotion | `Transforms/PromoteMemToReg.cpp` |
| Dead writes | `Transforms/DropDeadWrite.cpp` |
| Broadcast transfer | `Transforms/MergeIntoTransfer.cpp` |
| Register lowering | `Transforms/LowerToReg.cpp` |
| Group outlining/signature | `Transforms/OutlineVFGroup.cpp`, `Transforms/ShrinkVFSignature.cpp` |
| Group legalization/interfaces | `Transforms/LegalizeVFGroup.cpp`, `Transforms/AssignVFGroupTensors.cpp` |
| Loop-result reuse | `Transforms/BufferizeLoopResult.cpp` |
| Group reordering | `Transforms/ReorderGroupedOps.cpp` |
| Synchronization | `Transforms/InsertVFSync.cpp` |
| Register hoisting | `Transforms/HoistRegTensor.cpp` |
| Register reuse | `Transforms/ReuseRegTensor.cpp` |
| Ephemeral mask expansion | `Transforms/ExpandSimpleMaskReg.cpp` |
| Shared utilities | `lib/Dialect/AscVF2/Utils/Utils.cpp` |

Locations under `Transforms/` are relative to `experimental/asctile/lib/Dialect/AscVF2/`.

## Validation

```sh
cmake --build <build-dir> --target asctile-opt -j8
lit -sv experimental/asctile/test/Dialect/AscVF2
```

`lit` uses `ShTest`; the configured build's `bin/` and LLVM `FileCheck` must be on `PATH`. VF fusion requires C310.
For Python-facing pass changes, rebuild the extension through pip, for example
`python3 -m pip install --no-build-isolation -e . -v`; never invoke `setup.py` directly.

Look for end-to-end kernels in `experimental/asctile/python/test/target/` and its subdirectories. Prefer kernels with
an explicit `@asctile.jit(..., vf_fusion=2)` decorator. Always use `--compile-only`; run Python tests only when
debugging a specific case:

```sh
python3 -m pytest -vs experimental/asctile/python/test/target/test_softmax.py -k 'softmax_test_10-' --compile-only
```

Set `PYASC_PRINT_IR=1` only when explicitly dumping IR; it emits substantial output.
Set `PYASC_DUMP_PATH=<dir>` when durable compiler dumps are useful.
Use `CCACHE_DISABLE=1` before the CMake command when the sandbox or environment prevents ccache from working.

## Development Rules

- Preserve behavior when refactoring; run the AscVF2 lit suite before claiming completion.
- Add or update a focused FileCheck test for every lowering or fusion semantic change.
- Keep passes generic: reason about memory behavior and affine dependences, not named operations or one kernel's shape.
- Compare `vf_fusion=1` and `vf_fusion=2` dumps when investigating performance or correctness; final C++ can reveal
  load/store-count differences not obvious in MLIR.

## C++ Style

- Inline one-use helpers; define helpers primarily for genuine reuse.
- Remove unused functions and parameters unless overriding an API.
- Prefer short, recognizable names such as `srcBuf`, `dstBuf`, `hwWidth`, and `acc`; do not rename blindly.
- MLIR wrappers such as `Value`, `Type`, and `Attribute` have null states; use them instead of `optional` unless a
  third state is needed.
- In void functions, call side-effecting functions separately; do not return their result.
- Assume input IR is valid. Check only conditions that can invalidate IR or cause memory/system corruption.
- Use `TypeSwitch` for long `isa`/`dyn_cast` chains.
- Qualify `auto*` and `const auto&`; local by-value `const auto` is unnecessary.
- Merge `patterns.add` calls that share identical constructor arguments and pattern benefit; never merge different
  benefits.
- `clang-format` touched C++ files and keep behavior changes separate from refactors.
