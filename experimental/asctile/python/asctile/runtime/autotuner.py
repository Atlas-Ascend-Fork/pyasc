# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
import os
import pickle
import random
from typing import Any, Callable, Generic, List, Mapping, Optional, Tuple, Type, TypeVar, overload
from typing_extensions import ParamSpec, TypeAlias

from asc.lib.profiling import Profiler, task_time_median
from asc.lib.runtime import current_platform, is_model
from asc.runtime.cache import get_cache_manager, get_file_cache_key
from asc.runtime.launcher import MemoryOverflowError

from .jit import JITFunction

P = ParamSpec("P")
T = TypeVar("T")

Config: TypeAlias = Mapping[str, Any]


@dataclass(frozen=True)
class AutotuneResult:
    run_id: str
    config: Config
    task_time: float


class Autotuner(Generic[P, T]):
    cache_filename = "autotune.bin"

    def __init__(self, fn: JITFunction[P, T], *configs: Config, runs: int = 4, warmups: int = 1,
                 before_launch: Optional[Callable[[Config], None]] = None, profile_path: Optional[str] = None) -> None:
        self.fn = fn
        self.options = None
        self.configs = tuple(configs)
        if not self.configs:
            raise ValueError("At least one auto-tuning configuration must be provided")
        for config in configs:
            if not isinstance(config, Mapping):
                raise TypeError(f"All configs must be mappings, got {config.__class__.__name__}")
        self.runs = runs
        self.warmups = warmups
        self.results: List[AutotuneResult] = []
        self.before_launch = before_launch
        self.profiler = Profiler(profile_path)
        self.validate_configs()

    def __repr__(self) -> str:
        configs = ", ".join(map(repr, self.configs))
        return (
            f"{self.__class__.__name__}({self.fn.__name__}, {configs}, runs={self.runs}, warmups={self.warmups}, "
            f"before_launch={None if self.before_launch is None else ...}, profile_path={self.profiler.result_path!r})")

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        if self.results:
            return self.launch_best(*args, **kwargs)
        cached_result = self.load_cached_result(*args, **kwargs)
        if cached_result is not None:
            self.results = [cached_result]
            return self.launch_best(*args, **kwargs)
        for config in self.configs:
            fn_res, tune_res = self.launch_config(config, *args, **kwargs)
            self.results.append(tune_res)
        if all(math.isinf(result.task_time) for result in self.results):
            errors = "\n".join(result.run_id for result in self.results)
            self.prune_results(keep_best=False)
            raise RuntimeError(f"All auto-tuning configurations failed to launch:\n{errors}")
        self.results.sort(key=lambda r: r.task_time)
        if os.environ.get("PYASC_PRINT_AUTOTUNE"):
            print("Auto-tuning results:", self.fn.__name__, f"({self.warmups} warmups + {self.runs} runs)")
            col1_width = max(len(r.run_id) for r in self.results)
            col2_width = len(f"{self.results[-1].task_time:.2f}")
            for result in self.results:
                print(f"{result.run_id: <{col1_width}} | {result.task_time:{col2_width}.2f} μs | {result.config}")
        self.save_cached_result(*args, **kwargs)
        return fn_res

    def __getitem__(self, options):
        self.options = options
        return self

    @staticmethod
    def stable_repr(value: Any) -> str:
        value_repr = repr(value)
        if value_repr != repr(value):
            raise ValueError(f"Autotuner cache key contains a non-cacheable value: {value!r}")
        return value_repr

    @staticmethod
    def test_enabled() -> bool:
        return bool(os.environ.get("PYASC_TEST_AUTOTUNE"))

    @property
    def best_result(self) -> AutotuneResult:
        if not self.results:
            raise RuntimeError("No results were stored, maybe auto-tuning was never finished")
        return self.results[0]

    def launch_config(self, config: Config, *args: P.args, **kwargs: P.kwargs) -> Tuple[Optional[T], AutotuneResult]:
        if self.before_launch is not None:
            self.before_launch(config)
        launch_kwargs = kwargs | config
        try:
            result, run_id, task_time = self.benchmark(*args, **launch_kwargs)
        except MemoryOverflowError as error:
            return None, AutotuneResult(f"{error.__class__.__name__}: {error}", config, float("inf"))
        return result, AutotuneResult(run_id, config, task_time)

    def launch_best(self, *args: P.args, **kwargs: P.kwargs) -> T:
        config = self.results[0].config
        if self.before_launch is not None:
            self.before_launch(config)
        return self.fn[self.options](*args, **kwargs | config)

    def benchmark(self, *args, **kwargs) -> Tuple[T, str, float]:
        if self.test_enabled():
            result = self.fn[self.options](*args, **kwargs)
            return result, "test", random.random()
        if is_model():
            raise RuntimeError("Auto-tuning is only supported for NPU backend")
        with self.profiler.profile():
            for _ in range(self.runs + self.warmups):
                result = self.fn[self.options](*args, **kwargs)
        prof_result = self.profiler.last_result
        return result, prof_result.run_id, task_time_median(prof_result.tasks, skip=self.warmups)

    def prune_results(self, keep_best: bool = True) -> None:
        if not self.results:
            return
        if keep_best:
            del self.results[1:]
        else:
            self.results.clear()

    def pin_config(self, config: Config, prune: bool = False) -> None:
        if config not in self.configs:
            raise ValueError(f"Pinned configuration {config!r} is not one of the known configurations")
        self.results.insert(0, AutotuneResult("pinned", config, 0.0))
        if prune:
            self.prune_results(keep_best=True)

    def validate_configs(self) -> None:
        signature = inspect.signature(self.fn.fn)
        constexpr_args = self.fn.get_constexpr_args()
        for config in self.configs:
            for name in config:
                if name in signature.parameters and name not in constexpr_args:
                    raise ValueError(f"Runtime argument '{name}' is not allowed in the auto-tuning configuration")

    def get_cache_manager(self, *args: P.args, **kwargs: P.kwargs):
        launch_kwargs = kwargs | self.configs[0]
        prereqs, _ = self.fn.create_compile_prereqs(*args, **launch_kwargs)
        cache_factors = self.fn._gen_cache_factors(prereqs)
        invariants = self.stable_repr({
            "autotuner": 1,
            "configs": self.configs,
            "runs": self.runs,
            "warmups": self.warmups,
            "platform": current_platform(),
        })
        cache_key = get_file_cache_key(self.fn.cache_key, f"{cache_factors}__autotuner__{invariants}")
        return get_cache_manager(cache_key)

    def load_cached_result(self, *args: P.args, **kwargs: P.kwargs) -> Optional[AutotuneResult]:
        if self.before_launch is not None or self.test_enabled():
            return None
        cache_file = self.get_cache_manager(*args, **kwargs).get_file(self.cache_filename)
        if cache_file is None:
            return None
        try:
            with open(cache_file, "rb") as file:
                payload = pickle.load(file)
        except (OSError, pickle.PickleError, EOFError):
            return None
        if not isinstance(payload, dict):
            return None
        config = payload.get("best_config")
        if config not in self.configs:
            return None
        return AutotuneResult("cache", config, 0.0)

    def save_cached_result(self, *args: P.args, **kwargs: P.kwargs) -> None:
        if self.before_launch is not None or self.test_enabled():
            return
        payload = {"best_config": dict(self.best_result.config)}
        self.get_cache_manager(*args, **kwargs).put(pickle.dumps(payload), self.cache_filename)


@overload
def autotune(*configs: Config, runs: int = 4, warmups: int = 1) -> Callable[[JITFunction[P, T]], Autotuner[P, T]]:
    ...


def autotune(*configs: Config, cls: Type[Autotuner] = Autotuner, **kwargs):
    """
    Auto-tune a JIT kernel across a fixed set of constexpr argument values and JIT options.

    Each configuration is launched ``warmups + runs`` times. The first ``warmups`` launches are unmeasured, the
    following ``runs`` launches are measured, and their median task time is compared across configurations.
    The fastest configuration is used for the returned launch and persisted in the auto-tuner file cache.
    Subsequent launches reuse the cached configuration and do not benchmark the kernel again.

    Auto-tuning requires an active ``NPU`` backend; it is not supported on the ``Model`` backend.

    Args:
        configs: ``dict`` objects with candidate values as values. Keys may name arguments declared as ``ConstExpr``
            by the decorated JIT function or supported JIT options, e.g. ``reuse_alloc``.
        runs: Number of measured launches for each configuration. Their median task time determines the best
            configuration. Default is 4.
        warmups: Number of unmeasured launches performed before measurement. Default is 1.

    Returns:
        A decorator that accepts a JIT function and returns an auto-tuned kernel launcher.
        The launcher supports the same indexing syntax as a JIT function, e.g. ``kernel[core_num](...)``.

    Raises:
        TypeError: If the decorated object is not a JIT function.

    Examples:
        Declare the auto-tunable kernel with two tile sizes and compile options, then launch it with 8 blocks: ::

            @asctile.autotune(dict(tile_size=64, reuse_alloc=0), dict(tile_size=128, reuse_alloc=2))
            @asctile.jit
            def add_kernel(x_ptr, y_ptr, out_ptr, size: int, tile_size: asc.ConstExpr[int]):
                ...

            add_kernel[8](x, y, out, size)  # 'tile_size' is passed by the auto-tuner
    """

    def decorator(base):
        if isinstance(base, JITFunction):
            return cls(base, *configs, **kwargs)
        raise TypeError("asctile.autotune can only decorate JITFunction")

    return decorator
