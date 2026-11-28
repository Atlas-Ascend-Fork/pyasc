# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import itertools
import pickle

from asc.experimental import asctile
from asc.experimental.asctile.runtime import autotuner
from asc.experimental.asctile.runtime.jit import JITFunction
from asc.runtime import launcher
from asc.runtime import cache as runtime_cache
import pytest


class DisabledProfiler:

    def profile(self):
        raise AssertionError("profiling must not be enabled in test autotune mode")


@pytest.fixture(autouse=True)
def isolated_runtime_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_cache, "cache_options", runtime_cache.CacheOptions(dir=str(tmp_path)))
    monkeypatch.setattr(autotuner, "current_platform", lambda: "test-platform")


def constexpr_kernel(value: int, unroll_factor: asctile.ConstExpr):
    return value + unroll_factor


def test_autotuner_is_separate_from_jit_function_and_reuses_getitem():
    jit_function = JITFunction(constexpr_kernel, reuse_alloc=1)
    tuner = autotuner.Autotuner(jit_function, dict(unroll_factor=1))

    run = tuner[3]

    assert not isinstance(tuner, JITFunction)
    assert tuner.fn is jit_function
    assert run is tuner


def test_config_rejects_runtime_arguments():
    with pytest.raises(ValueError, match="Runtime argument 'value'"):
        autotuner.Autotuner(JITFunction(constexpr_kernel), dict(value=1))


def test_config_does_not_validate_unknown_names():
    tuner = autotuner.Autotuner(JITFunction(constexpr_kernel), dict(unknown=1))

    assert dict(unknown=1) in tuner.configs


def test_jit_function_exposes_constexpr_argument_names():
    jit_function = JITFunction(constexpr_kernel)

    assert jit_function.get_constexpr_args() == {"unroll_factor"}


def test_autotuner_cache_key_is_stable_and_input_sensitive():
    jit_function = JITFunction(constexpr_kernel, reuse_alloc=1)
    other_jit_function = JITFunction(constexpr_kernel, reuse_alloc=2)
    tuner = autotuner.Autotuner(jit_function, dict(unroll_factor=1))
    other_tuner = autotuner.Autotuner(other_jit_function, dict(unroll_factor=1))

    cache_key = tuner.get_cache_manager(11).key

    assert cache_key == tuner.get_cache_manager(11).key
    assert cache_key != tuner.get_cache_manager(12.0).key
    assert cache_key != other_tuner.get_cache_manager(11).key


def test_autotuner_cache_key_accepts_config_only_constexpr_arguments():
    jit_function = JITFunction(constexpr_kernel)
    tuner = autotuner.Autotuner(jit_function, dict(unroll_factor=1))

    cache_key = tuner.get_cache_manager(11).key

    assert cache_key == tuner.get_cache_manager(11).key


def test_test_mode_runs_once_without_backend_check_or_profiling(monkeypatch):
    calls = []

    def fake_run(self, launch_options, args, kwargs):
        calls.append((args, kwargs))
        return "result"

    monkeypatch.setattr(JITFunction, "_run_with_options", fake_run)
    tuner = autotuner.Autotuner(JITFunction(constexpr_kernel), dict(unroll_factor=1), runs=4, warmups=2)
    tuner.profiler = DisabledProfiler()

    def fail_backend_check():
        raise AssertionError("backend check must be skipped in test autotune mode")

    monkeypatch.setenv("PYASC_TEST_AUTOTUNE", "1")
    monkeypatch.setattr(autotuner, "is_model", fail_backend_check)

    result, run_id, task_time = tuner.benchmark(11, unroll_factor=1)

    assert result == "result"
    assert calls == [((11, ), dict(unroll_factor=1))]
    assert run_id == "test"
    assert isinstance(task_time, float)


def test_memory_overflow_error_is_runtime_error(monkeypatch):
    monkeypatch.setattr(launcher.rt, "get_soc_version", lambda: "test-platform")
    monkeypatch.setattr(
        launcher, "get_platform_info",
        lambda platform: launcher.PlatformInfo(ub_size=0, l1_size=0, l0a_size=0, l0b_size=0, l0c_size=0, bt_size=0))

    assert issubclass(launcher.MemoryOverflowError, RuntimeError)
    with pytest.raises(launcher.MemoryOverflowError, match="UB overflow"):
        launcher.Launcher.check_memory_overflow({"UB": 1})


def test_autotuner_continues_after_memory_overflow(monkeypatch):
    monkeypatch.delenv("PYASC_TEST_AUTOTUNE", raising=False)

    def fake_benchmark(self, *args, **kwargs):
        if kwargs["unroll_factor"] == 1:
            raise launcher.MemoryOverflowError("UB overflow")
        return "result", "run", 1.0

    monkeypatch.setattr(autotuner.Autotuner, "benchmark", fake_benchmark)
    tuner = autotuner.Autotuner(
        JITFunction(constexpr_kernel),
        dict(unroll_factor=1),
        dict(unroll_factor=2),
    )

    result = tuner(11)

    assert result == "result"
    assert tuner.best_result.config == dict(unroll_factor=2)
    assert tuner.results[-1].task_time == float("inf")
    assert tuner.results[-1].run_id == "MemoryOverflowError: UB overflow"


def test_autotuner_raises_and_prunes_when_all_launches_overflow(monkeypatch):
    monkeypatch.delenv("PYASC_TEST_AUTOTUNE", raising=False)

    def fake_benchmark(self, *args, **kwargs):
        raise launcher.MemoryOverflowError(f"{kwargs['unroll_factor']} overflow")

    monkeypatch.setattr(autotuner.Autotuner, "benchmark", fake_benchmark)
    tuner = autotuner.Autotuner(
        JITFunction(constexpr_kernel),
        dict(unroll_factor=1),
        dict(unroll_factor=2),
    )

    with pytest.raises(
            RuntimeError, match=("All auto-tuning configurations failed to launch:\n"
                                 "MemoryOverflowError: 1 overflow\n"
                                 "MemoryOverflowError: 2 overflow")):
        tuner(11)

    assert tuner.results == []


def test_pin_config_inserts_existing_configuration_as_best():
    pinned = dict(unroll_factor=2)
    tuner = autotuner.Autotuner(JITFunction(constexpr_kernel), dict(unroll_factor=1), pinned, dict(unroll_factor=4))

    tuner.pin_config(pinned)

    assert tuner.best_result.config is pinned
    assert tuner.best_result.task_time == 0.0
    assert len(tuner.results) == 1


def test_pin_config_rejects_unknown_configuration():
    tuner = autotuner.Autotuner(JITFunction(constexpr_kernel), dict(unroll_factor=1))

    with pytest.raises(ValueError, match="not one of the known"):
        tuner.pin_config(dict(unroll_factor=2))


def test_prune_keeps_only_best_result_after_tuning(monkeypatch):
    timings = [3.0, 1.0, 2.0]

    def fake_benchmark(self, *args, **kwargs):
        return "result", f"run-{len(self.results)}", timings[len(self.results)]

    monkeypatch.setattr(autotuner.Autotuner, "benchmark", fake_benchmark)
    tuner = autotuner.Autotuner(
        JITFunction(constexpr_kernel),
        dict(unroll_factor=1),
        dict(unroll_factor=2),
        dict(unroll_factor=4),
    )

    tuner(11, unroll_factor=1)
    tuner.pin_config(dict(unroll_factor=2), prune=True)

    assert [result.config for result in tuner.results] == [dict(unroll_factor=2)]
    assert tuner.best_result.task_time == 0.0


def test_autotuner_cache_hit_launches_best_without_benchmarking(monkeypatch):
    monkeypatch.delenv("PYASC_TEST_AUTOTUNE", raising=False)
    launches = []

    def fake_run(self, launch_options, args, kwargs):
        launches.append(kwargs)
        return "launch"

    benchmark_calls = []

    def fake_benchmark(self, *args, **kwargs):
        benchmark_calls.append(kwargs)
        task_time = 1.0 if kwargs["unroll_factor"] == 2 else 3.0
        return "benchmark", f"run-{len(benchmark_calls)}", task_time

    monkeypatch.setattr(JITFunction, "_run_with_options", fake_run)
    monkeypatch.setattr(autotuner.Autotuner, "benchmark", fake_benchmark)
    jit_function = JITFunction(constexpr_kernel)
    configs = (dict(unroll_factor=1), dict(unroll_factor=2), dict(unroll_factor=4))
    tuner = autotuner.Autotuner(jit_function, *configs)

    tuner(11, unroll_factor=1)
    assert tuner.best_result.config == dict(unroll_factor=2)

    benchmark_calls.clear()
    launches.clear()
    cached_tuner = autotuner.Autotuner(jit_function, *configs)
    result = cached_tuner(11, unroll_factor=1)

    assert result == "launch"
    assert benchmark_calls == []
    assert launches == [dict(unroll_factor=2)]
    assert cached_tuner.best_result.config == dict(unroll_factor=2)


def test_before_launch_disables_autotuner_cache(monkeypatch):
    monkeypatch.delenv("PYASC_TEST_AUTOTUNE", raising=False)
    benchmark_calls = []

    def fake_benchmark(self, *args, **kwargs):
        benchmark_calls.append(kwargs)
        return "result", f"run-{len(benchmark_calls)}", 1.0

    monkeypatch.setattr(autotuner.Autotuner, "benchmark", fake_benchmark)
    jit_function = JITFunction(constexpr_kernel)
    before_launch = lambda config: None
    configs = (dict(unroll_factor=1), dict(unroll_factor=2))

    autotuner.Autotuner(jit_function, *configs, before_launch=before_launch)(11)
    autotuner.Autotuner(jit_function, *configs, before_launch=before_launch)(11)

    assert len(benchmark_calls) == len(configs) * 2


def test_autotuner_cache_filename_is_class_constant():
    assert autotuner.Autotuner.cache_filename == "autotune.bin"


def test_autotuner_cache_payload_stores_only_best_config(monkeypatch):
    monkeypatch.delenv("PYASC_TEST_AUTOTUNE", raising=False)

    def fake_benchmark(self, *args, **kwargs):
        task_time = 1.0 if kwargs["unroll_factor"] == 2 else 3.0
        return "result", "run", task_time

    monkeypatch.setattr(autotuner.Autotuner, "benchmark", fake_benchmark)
    jit_function = JITFunction(constexpr_kernel)
    configs = (dict(unroll_factor=1), dict(unroll_factor=2))
    tuner = autotuner.Autotuner(jit_function, *configs)

    tuner(11)
    cache_file = tuner.get_cache_manager(11).get_file(tuner.cache_filename)
    with open(cache_file, "rb") as file:
        payload = pickle.load(file)

    assert payload == {"best_config": dict(unroll_factor=2)}


def test_autotuner_cache_key_changes_with_configs(monkeypatch):
    monkeypatch.delenv("PYASC_TEST_AUTOTUNE", raising=False)
    benchmark_calls = []

    def fake_benchmark(self, *args, **kwargs):
        benchmark_calls.append(kwargs)
        return "benchmark", f"run-{len(benchmark_calls)}", 1.0

    monkeypatch.setattr(autotuner.Autotuner, "benchmark", fake_benchmark)
    jit_function = JITFunction(constexpr_kernel)
    autotuner.Autotuner(jit_function, dict(unroll_factor=1))(11, unroll_factor=1)

    benchmark_calls.clear()
    autotuner.Autotuner(jit_function, dict(unroll_factor=2))(11, unroll_factor=1)

    assert len(benchmark_calls) == 1


def test_autotuner_cache_key_includes_complete_config_set(monkeypatch):
    monkeypatch.delenv("PYASC_TEST_AUTOTUNE", raising=False)
    benchmark_calls = []

    def fake_benchmark(self, *args, **kwargs):
        benchmark_calls.append(kwargs)
        return "result", f"run-{len(benchmark_calls)}", 1.0

    monkeypatch.setattr(autotuner.Autotuner, "benchmark", fake_benchmark)
    jit_function = JITFunction(constexpr_kernel)
    autotuner.Autotuner(
        jit_function,
        dict(unroll_factor=1),
        dict(unroll_factor=2),
    )(11)

    benchmark_calls.clear()
    autotuner.Autotuner(
        jit_function,
        dict(unroll_factor=1),
        dict(unroll_factor=4),
    )(11)

    assert len(benchmark_calls) == 2


def test_autotuner_rejects_non_cacheable_config(monkeypatch):
    monkeypatch.delenv("PYASC_TEST_AUTOTUNE", raising=False)
    repr_numbers = itertools.count()

    class UnstableConfig(dict):

        def __repr__(self):
            return f"UnstableConfig({next(repr_numbers)})"

    jit_function = JITFunction(constexpr_kernel)
    tuner = autotuner.Autotuner(jit_function, UnstableConfig(unroll_factor=1))

    with pytest.raises(ValueError, match="configs.*UnstableConfig\\(2\\)"):
        tuner(11, unroll_factor=1)
