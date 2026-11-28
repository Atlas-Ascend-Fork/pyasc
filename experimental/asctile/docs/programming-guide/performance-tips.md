<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Performance tips

```{contents} Table of Contents
:local:
```

## Key Performance Principles

### 1. More Cores = More Performance

Ascend NPUs expose dozens of AI Cores (up to 56+ on C310). **Distribute work across as many cores as possible.** Under-utilizing cores is the single biggest performance mistake.

```python
# Good: spread rows across all available cores
rows_per_block = asctile.ceildiv(input_num_rows, asctile.block_num())
block_offset = asctile.block_idx() * rows_per_block

# Bad: only use a few cores, leaving most idle
if asctile.block_idx() < 4:
    ...
```

Launch with the maximum core count the problem allows:
```python
kernel[56](...)  # use all 56 cores when workload is large enough
```

Each core processes its own slice of data in parallel — doubling active cores roughly doubles throughput.

### 2. Bigger Tiles = Better Performance, Less Transfer Overhead

Every `asctile.copy_in` / `asctile.copy_out` is a DMA transfer between global memory (HBM) and on-chip memory. **Larger tiles amortize transfer latency and reduce the total number of transfers.**

```python
# Good: large tile, fewer transfers
tile_length = 10496  # ~40 KB for fp32 — fills UB well
for i in asctile.range(loop_count, ...):
    xt = asctile.copy_in(in_gm, [tile_length], ...)

# Bad: small tile, many transfers, high overhead
tile_length = 128  # only 512 bytes — mostly waiting on DMA
for i in asctile.range(huge_loop_count, ...):
    xt = asctile.copy_in(in_gm, [tile_length], ...)
```

**Rule of thumb**: make tiles as large as UB memory allows (typically ~256 KB total, shared among all live tiles). The test suite uses tiles of 10496–18752 elements for fp32 kernels (40–75 KB per tile).

### 3. Use `ConstExpr` for Static Code Generation

**All tiling parameters, shapes, and loop bounds must be typed as `asctile.ConstExpr`.** This tells the JIT compiler to treat them as compile-time constants, enabling:
- Full loop unrolling (`unroll_factor` baked into generated code)
- Static tensor allocation (`static_alloc=True`) — no runtime memory management overhead
- Dead-code elimination and constant folding in the MLIR pipeline

```python
# Good: all tiling params are ConstExpr → static, optimized code
@asctile.jit(static_alloc=True, reuse_alloc=2)
def kernel(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress,
           input_length: asctile.ConstExpr, tile_length: asctile.ConstExpr,
           unroll_factor: asctile.ConstExpr):
    ...

# Bad: plain int → dynamic code, no unrolling, runtime overhead
def kernel(input_ptr, output_ptr, input_length: int, tile_length: int):
    ...
```

`ConstExpr` values are fixed at JIT compile time. Changing them triggers recompilation, but the resulting binary is significantly faster.

---

## JIT Options and Loop Control

### JIT Decorator Options

| Option | Purpose | Recommended Usage |
|--------|---------|-------------------|
| `static_alloc` | Static tensor allocation at compile time | Enabling recommended for most kernels. Leads to faster execution, no runtime memory management. Requires all local tensor shapes to be `ConstExpr`. When disabled provides TPipe-backed (dynamic) allocation. Use only for complex control flow with variable tile sizes. Slower but more flexible. |
| `reuse_alloc` | Reuse freed memory regions across iterations | Setting to mode `2` recommended for most kernels. Reduces peak on-chip memory consumption by reusing buffers. |
| `vf_fusion` | Enable vector fusion (experimental) | Advanced option, when enabled lowering generated code to register level API. Experimental feature. May improve performance for element-wise chains. |
| `always_compile` | Cached kernels usage | When enabled ignores cached kernels, provides recompilation. |

### Loop Control: `unroll_factor` and `parallel`

These parameters control how loops are compiled and executed:

**`unroll_factor: int`** (default is 1, passed to `asctile.range()` or `range()`):
- Unrolls the loop by the given factor at compile time
- Reduces loop overhead and enables instruction-level parallelism
- **Recommended**: `unroll_factor=2` for most kernels, `unroll_factor=1` for very large tiles or memory-bound operations
- Must be `ConstExpr` for static unrolling

**`gm_barrier: bool`** (default is `False`, passed to `asctile.range()` or `range()`):
- Inserts barriers for data transfer pipes and disables parallel load/store optimization across loop iterations
- Only necessary when different iterations write and the read from the same memory
- When `False`, allows overlapping DMA transfers with computation
- **Recommended: do not enable** for loops that perform independent tile operations

```python
# Recommended pattern for outer tile loop
for i in asctile.range(loop_count, unroll_factor=2):
    xt = asctile.copy_in(in_gm, [tile_length], ...)
    zt = xt + yt
    asctile.copy_out(zt, out_gm, ...)
```

---

## Vector function (VF) block fusion

For a sequence of basic elementwise or reduction vector operations, PyAsc can automatically fuse them into a single **VF (vector function)** block that executes at the register level, avoiding redundant UB reads/writes between intermediate steps. Enable this by passing `vf_fusion=True` to the JIT decorator:

```python
@asctile.jit(vf_fusion=True)
def kernel(x_ptr, y_ptr, out_ptr, size: int, tile_size: asctile.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    for i in asctile.range(asctile.ceildiv(size, tile_size)):
        x = asctile.copy_in(x_gm, [i * tile_size], [tile_size])
        y = asctile.copy_in(y_gm, [i * tile_size], [tile_size])
        # These elementwise ops are fused into a single VF block
        result = (x + y) * x - y
        asctile.copy_out(result, out_gm, [i * tile_size])
```

The automatic fusion handles straightforward chains of built-in arithmetic and reduction operations. For more complex patterns — such as custom register-level logic, specialized masking, or operations not expressible through the standard API — use {py:func}`asctile.inline_vf` to embed raw Ascend C register code directly as a VF block:

```python
# x * y + z — three inputs, processed in 64-element vector chunks
out = asctile.inline_vf(
    """
    auto* out_ptr = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());
    auto* x_ptr = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());
    auto* y_ptr = reinterpret_cast<__ubuf__ float*>($2.GetPhyAddr());
    auto* z_ptr = reinterpret_cast<__ubuf__ float*>($3.GetPhyAddr());
    AscendC::Reg::RegTensor<float> x_reg, y_reg, z_reg, xy_reg, result_reg;
    uint32_t count = 256;
    for (uint16_t i = 0; i < 4; i += 1) {
      uint32_t offset = i * 64;
      AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(count);
      AscendC::Reg::DataCopy(x_reg, x_ptr + offset);
      AscendC::Reg::DataCopy(y_reg, y_ptr + offset);
      AscendC::Reg::Mul(xy_reg, x_reg, y_reg, mask);
      AscendC::Reg::DataCopy(z_reg, z_ptr + offset);
      AscendC::Reg::Add(result_reg, xy_reg, z_reg, mask);
      AscendC::Reg::DataCopy(out_ptr + offset, result_reg, mask);
    }
    """,
    shape=x.shape, dtype=x.dtype, inputs=[x, y, z])
```

Inside the code string, `$0` refers to the output tensor and `$1`, `$2`, ... refer to input tensors in the order they appear in the `inputs` list. All input tensors must reside in UB memory.

---

## Measuring Performance with the Profiler

The test suite provides a built-in `profiler` fixture that wraps kernel launches with NPU hardware profiling.

### Enabling Profiling

Profiling only activates on real NPU hardware with the `--profile` flag:

```bash
# Profile on NPU backend (requires physical Ascend NPU)
pytest --backend NPU --profile python/test/asctile/target/test_vadd.py

# Profile with multiple runs per test (more stable median)
pytest --backend NPU --profile --runs 10 python/test/asctile/target/test_vadd.py

# Select specific platform
pytest --backend NPU --profile --platform Ascend950PR_9599 python/test/asctile/target/
```

Without `--profile` or when not using the NPU backend, a `StubProfiler` is used (no-op).

### Using the Profiler in Tests

Wrap the kernel launch loop with `profiler.profile()`:

```python
def test_my_kernel(profiler, runs, block_num, ...):
    in_tensor = torch.randn(...)
    out_tensor = torch.zeros(...)

    with profiler.profile():
        for _ in range(runs):
            my_kernel[block_num](in_tensor, out_tensor, ...)

    expected = torch_ref(in_tensor)
    torch.testing.assert_close(out_tensor, expected, atol=1e-3, rtol=1e-3)
```

The `runs` fixture (controlled by `--runs`, default=1) determines how many kernel launches are timed. Multiple runs produce a more stable median duration.

### Output

After the test session, profiling results are printed in a summary table:

```
============================== Profiling results ==============================
python/test/asctile/target/test_vadd.py::test_add[...]: 12.34 μs
python/test/asctile/target/test_softmax.py::test_softmax[...]: 45.67 μs
```

The reported duration is the **median task time** across all runs (first run is skipped as warmup via `skip=1` in `task_time_median`).

### Saving Profiling Results

By default, profiling data is stored in a temporary directory. To persist the raw CSV reports, use the `--profile-path` option:

```bash
pytest --backend NPU --profile --profile-path ./profiling_results test.py
```

### Profiling Tips

- **Use `--runs 10` or more** for stable measurements — single runs have high variance.
- **Vary core number and tile sizes**: run the same test with different `tile_length` values to see transfer overhead impact. Vary `block_num` to find the sweet spot (more cores = more parallelism, but decrease if work per core is too small).
- **Use `always_compile=False` for fast iterations**: by default, compiled kernels are cached. Set `always_compile=True` only when actively debugging compilation issues. Use `--compile-only` to verify tiling decisions compile without UB overflow before running on hardware.
