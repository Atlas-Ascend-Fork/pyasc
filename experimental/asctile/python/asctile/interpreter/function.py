# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from collections.abc import Iterator
from contextlib import contextmanager
import inspect
import types
from typing import Any, Dict, Generator, Set, Tuple

import numpy as np

from asc.codegen.specialization import Specialization
from asc.language.core.constexpr import ConstExpr
from asc.runtime.cache import get_mem_cache_key
from asc.runtime.jit import CompilePrereqs
from asc.runtime.launcher import LaunchOptions

from ..runtime.jit import JITFunction
from .operations import block_context, join_stage, range as asctile_range
from .patch import make_asc_module, make_module, to_interpreted
from .types import GlobalAddress, dtype_to_numpy


def wrap_argument(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return GlobalAddress(value)
    if isinstance(value, (int, float, np.generic)):
        return value
    try:
        import torch
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                return value.item()
            name = str(value.dtype).removeprefix("torch.")
            target = dtype_to_numpy(name)
            uint = {1: torch.uint8, 2: torch.uint16, 4: torch.uint32, 8: torch.uint64}[target.itemsize]
            return GlobalAddress(value.view(uint).numpy(force=True).view(target))
    except ModuleNotFoundError:
        pass
    raise TypeError(f"{value.__class__.__name__} arguments are not supported by the interpreter")


@contextmanager
def interpret_asctile(fn: types.FunctionType) -> Generator[None, None, None]:
    namespace = fn.__globals__
    absent = object()
    names = {"asctile": make_module(), "range": asctile_range}
    if "asc" in namespace:
        names["asc"] = make_asc_module()
    original = {name: namespace.get(name, absent) for name in names}
    namespace.update(names)
    try:
        yield
    finally:
        for name, value in original.items():
            if value is absent:
                namespace.pop(name, None)
            else:
                namespace[name] = value


class InterpretedFunction(JITFunction):

    def __init__(self, fn, **options):
        super().__init__(fn, **options)
        self._validated: Set[str] = set()
        self.sub_block_num = 1

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with interpret_asctile(self.fn):
            return self.fn(*args, **kwargs)

    def _compile_and_cache(self, prereqs: CompilePrereqs) -> Tuple[Dict[str, ConstExpr], None]:
        if prereqs.compile_options.cv_ratio not in (1, 2):
            raise RuntimeError(f"'cv_ratio' is only allowed to be 1 or 2, got {prereqs.compile_options.cv_ratio}")
        self.sub_block_num = prereqs.compile_options.cv_ratio
        for name, expr in prereqs.constexprs.items():
            if isinstance(expr.value, Iterator):
                prereqs.constexprs[name] = ConstExpr(tuple(expr.value))
        cache_key = get_mem_cache_key(self._gen_cache_factors(prereqs))
        if prereqs.compile_options.always_compile or cache_key not in self._validated:
            self._validate(prereqs)
            self._validated.add(cache_key)
        return prereqs.constexprs, None

    def _validate(self, prereqs: CompilePrereqs) -> None:
        self._run_codegen(Specialization(prereqs.arg_types, prereqs.constexprs), prereqs.codegen_options)

    def _run_launcher(self, constexprs: Dict[str, ConstExpr], options: LaunchOptions, runtime_args: Tuple[Any],
                      mem_cache_key: Any) -> None:
        del mem_cache_key
        block_num = options.core_num
        if block_num is None:
            raise NotImplementedError("'core_num' must be set explicitly in interpreter mode")
        if block_num <= 0:
            raise ValueError("'core_num' must be positive")
        arg_values = map(wrap_argument, runtime_args)
        fn_args = tuple(
            to_interpreted(ConstExpr.unwrap(constexprs[name])) if name in constexprs else next(arg_values)
            for name in inspect.signature(self.fn).parameters)
        with interpret_asctile(self.fn):
            join_stage.clear()
            for block_idx in range(block_num):
                for sub_block_idx in range(self.sub_block_num):
                    block_context.set(block_idx, block_num, sub_block_idx, self.sub_block_num)
                    self.fn(*fn_args)
