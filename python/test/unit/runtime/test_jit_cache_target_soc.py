# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import hashlib
import io
from dataclasses import dataclass
from types import MethodType, SimpleNamespace

import pytest

from asc.runtime import config
from asc.runtime import jit as jit_module


@dataclass
class _Options:
    marker: int = 0


@dataclass
class _CompiledKernel:
    compiled_for: str


class _FileCache:
    files = {}

    def __init__(self, key):
        self.key = key

    def get_file(self, filename):
        key = f"/virtual/{self.key}/{filename}"
        return key if key in self.files else None

    def put(self, data, filename):
        self.files[f"/virtual/{self.key}/{filename}"] = data


def _make_jit():
    jit = SimpleNamespace(
        fn=SimpleNamespace(__name__="test_kernel"),
        fn_name="test_kernel",
        cache_key="same-function-source",
        kernel_cache={},
        compile_count=0,
        get_arg_dtype=lambda _: "int32",
    )
    jit._gen_cache_factors = MethodType(jit_module.JITFunction._gen_cache_factors, jit)
    jit._cache_kernel = MethodType(jit_module.JITFunction._cache_kernel, jit)

    def compile_kernel(self, _prereqs):
        self.compile_count += 1
        return _CompiledKernel(jit_module.rt.get_soc_version().value)

    jit._compile_kernel = MethodType(compile_kernel, jit)
    return jit


@pytest.fixture
def cache_runtime(monkeypatch):
    _FileCache.files = {}
    monkeypatch.setattr(jit_module, "get_mem_cache_key", lambda value: hashlib.sha256(value.encode()).hexdigest())
    monkeypatch.setattr(
        jit_module,
        "get_file_cache_key",
        lambda fn_key, value: hashlib.sha256(f"{fn_key}:{value}".encode()).hexdigest(),
    )
    monkeypatch.setattr(jit_module, "get_cache_manager", _FileCache)
    monkeypatch.setattr(jit_module, "open", lambda path, _mode: io.BytesIO(_FileCache.files[str(path)]), raising=False)
    monkeypatch.setattr(jit_module.rt, "use_model", lambda: None)
    monkeypatch.setattr(jit_module.rt, "is_available", lambda: True)

    def set_soc(soc):
        current_soc[0] = soc

    current_soc = [config.Platform.Ascend910B3]
    monkeypatch.setattr(jit_module.rt, "get_soc_version", lambda: current_soc[0])
    monkeypatch.setattr(jit_module.rt, "set_soc_version", set_soc)
    return current_soc


def _prereqs():
    return SimpleNamespace(
        arg_types={},
        constexprs={},
        codegen_options=_Options(),
        compile_options=_Options(),
    )


def _set_platform(soc):
    config.set_platform(config.Backend.Model, soc, check=False)


@pytest.mark.parametrize(
    ("first_soc", "other_soc"),
    [
        (config.Platform.Ascend910B3, config.Platform.Ascend950PR_9599),
        (config.Platform.Ascend950PR_9599, config.Platform.Ascend910B3),
        (config.Platform.Ascend910B3, config.Platform.Ascend910B4),
        (config.Platform.Ascend910B4, config.Platform.Ascend910B3),
    ],
)
def test_memory_cache_is_partitioned_by_target_soc(cache_runtime, first_soc, other_soc):
    jit = _make_jit()
    _set_platform(first_soc)
    first, first_key = jit._cache_kernel(_prereqs())
    same, same_key = jit._cache_kernel(_prereqs())
    _set_platform(other_soc)
    other, other_key = jit._cache_kernel(_prereqs())
    _set_platform(first_soc)
    restored, restored_key = jit._cache_kernel(_prereqs())

    assert first.compiled_for == same.compiled_for == restored.compiled_for == first_soc.value
    assert other.compiled_for == other_soc.value
    assert first_key == same_key == restored_key
    assert other_key != first_key
    assert jit.compile_count == 2


@pytest.mark.parametrize(
    ("first_soc", "other_soc"),
    [
        (config.Platform.Ascend910B3, config.Platform.Ascend950PR_9599),
        (config.Platform.Ascend950PR_9599, config.Platform.Ascend910B3),
        (config.Platform.Ascend910B3, config.Platform.Ascend910B4),
        (config.Platform.Ascend910B4, config.Platform.Ascend910B3),
    ],
)
def test_file_cache_is_partitioned_across_jit_instances(cache_runtime, first_soc, other_soc):
    outputs = []
    compile_counts = []
    keys = []
    for soc in (first_soc, first_soc, other_soc, first_soc):
        _set_platform(soc)
        jit = _make_jit()
        result, key = jit._cache_kernel(_prereqs())
        outputs.append(result.compiled_for)
        compile_counts.append(jit.compile_count)
        keys.append(key)

    assert outputs == [first_soc.value, first_soc.value, other_soc.value, first_soc.value]
    assert compile_counts == [1, 0, 1, 0]
    assert keys[0] == keys[1] == keys[3]
    assert keys[2] != keys[0]
