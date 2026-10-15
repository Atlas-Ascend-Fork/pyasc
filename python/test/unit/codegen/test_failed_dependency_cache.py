# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
import importlib.util
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest


def load_module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def function_class(monkeypatch):
    # Load the actual Host modules without initializing the native asc package.
    root = Path(__file__).resolve().parents[3] / "asc"
    for suffix in ("", ".common", ".language", ".language.core", ".codegen"):
        name = "asc" + suffix
        module = types.ModuleType(name)
        module.__path__ = [str(root.joinpath(*suffix.strip(".").split(".")))]
        monkeypatch.setitem(sys.modules, name, module)
    load_module(monkeypatch, "asc.common.compat", root / "common/compat.py")
    load_module(monkeypatch, "asc.language.core.constexpr", root / "language/core/constexpr.py")
    load_module(monkeypatch, "asc.codegen.name_scope", root / "codegen/name_scope.py")
    utils_path = root / "language/core/utils.py"
    tree = ast.parse(utils_path.read_text())
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "jit_allowed") or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "JIT_INTERNAL" for target in node.targets))
    ]
    utils = types.ModuleType("asc.language.core.utils")
    utils.Any = Any
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(utils_path), "exec"), utils.__dict__)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    function = load_module(monkeypatch, "asc.codegen.function", root / "codegen/function.py")
    load_module(monkeypatch, "asc.codegen.dependencies_finder", root / "codegen/dependencies_finder.py")
    return function.Function


def kernel_module(monkeypatch, tmp_path, source):
    path = tmp_path / "cache_retry_inputs.py"
    path.write_text(source)
    return load_module(monkeypatch, "cache_retry_inputs", path)


def test_repeated_invalid_dependency_still_raises(function_class, monkeypatch, tmp_path):
    module = kernel_module(monkeypatch, tmp_path, "def dependency():\n    pass\n\ndef kernel():\n    dependency()\n")
    function = function_class(module.kernel)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="Unsupported function referenced"):
            _ = function.cache_key


def test_corrected_dependency_is_analyzed_on_retry(function_class, monkeypatch, tmp_path):
    module = kernel_module(monkeypatch, tmp_path,
                           "def dependency():\n    pass\n\ndef kernel():\n    value = dependency\n")
    function = function_class(module.kernel)
    with pytest.raises(RuntimeError, match="Unsupported function referenced"):
        _ = function.cache_key
    module.dependency = 7
    retried = function.cache_key
    assert re.fullmatch(r"[0-9a-f]{64}", retried)
    assert retried == function_class(module.kernel).cache_key
    assert any(value == 7 for value, _ in function.used_global_vals.values())


def test_transitive_failure_still_raises_on_retry(function_class, monkeypatch, tmp_path):
    module = kernel_module(
        monkeypatch, tmp_path,
        "def dependency():\n    pass\n\ndef child():\n    dependency()\n\ndef parent():\n    child()\n")
    module.child = function_class(module.child)
    parent = function_class(module.parent)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Unsupported function referenced"):
            _ = parent.cache_key


def test_successful_key_remains_memoized(function_class, monkeypatch, tmp_path):
    module = kernel_module(monkeypatch, tmp_path, "value = 3\n\ndef kernel():\n    result = value\n")
    function = function_class(module.kernel)
    key = function.cache_key
    assert re.fullmatch(r"[0-9a-f]{64}", key)
    monkeypatch.setattr(function, "parse", lambda: pytest.fail("memoized key was recomputed"))
    assert function.cache_key == key


def test_recursive_dependency_still_terminates(function_class, monkeypatch, tmp_path):
    module = kernel_module(monkeypatch, tmp_path, "def kernel():\n    kernel()\n")
    module.kernel = function_class(module.kernel)
    key = module.kernel.cache_key
    assert re.fullmatch(r"[0-9a-f]{64}", key)
    assert module.kernel.cache_key == key
