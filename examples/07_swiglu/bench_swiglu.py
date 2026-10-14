# Copyright (c) 2026 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import subprocess
from pathlib import Path

import torch

import asc.runtime.config as config
import asc.lib.runtime as rt
from swiglu import swiglu_kernel, compute_launch_params

ASCENDC_DEMO = Path(__file__).resolve().parent / "ascendc" / "build" / "demo"


def run_pyasc(shape, warmup, iters):
    total = shape[0] * shape[1] * shape[2]
    cores, block_len, tile_len = compute_launch_params(total, 4)
    padded = cores * block_len
    fused = torch.empty(padded * 2, dtype=torch.float32, device="npu")
    y_pad = torch.empty(padded, dtype=torch.float32, device="npu")
    for _ in range(warmup + iters):
        swiglu_kernel[cores, rt.current_stream()](fused, y_pad, block_len, tile_len, padded)
    torch.npu.synchronize()


def run_ascendc(shape, warmup, iters):
    cmd = [
        str(ASCENDC_DEMO), "--batch",
        str(shape[0]), "--seq",
        str(shape[1]), "--hidden",
        str(shape[2]), "--warmup",
        str(warmup), "--iters",
        str(iters)
    ]
    subprocess.run(cmd, check=True)


def run_torch_npu(shape, warmup, iters):
    import torch_npu
    x = torch.empty((shape[0], shape[1], shape[2] * 2), dtype=torch.float32, device="npu")
    for _ in range(warmup + iters):
        torch_npu.npu_swiglu(x, dim=-1)
    torch.npu.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pyasc", "ascendc", "torch_npu"], default="pyasc")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("-r", type=str, default="NPU")
    parser.add_argument("-v", type=str, default="Ascend910B4")
    args = parser.parse_args()

    config.set_platform(config.Backend(args.r), config.Platform(args.v))
    torch.manual_seed(20260621)
    shape = (args.batch, args.seq, args.hidden)
    if args.backend == "pyasc":
        run_pyasc(shape, args.warmup, args.iters)
    elif args.backend == "ascendc":
        run_ascendc(shape, args.warmup, args.iters)
    else:
        run_torch_npu(shape, args.warmup, args.iters)
    print(f"DONE op=swiglu backend={args.backend} shape={shape} warmup={args.warmup} iters={args.iters}")


if __name__ == "__main__":
    main()
