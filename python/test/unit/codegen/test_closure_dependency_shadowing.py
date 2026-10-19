# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Host-only dependency hashes must follow Python closure name precedence."""
import ast
import importlib.util
from pathlib import Path
import sys
import types
from typing import Any

import pytest


def value():
    return -1


def make_kernel(captured, function_class):
    value = captured

    def kernel():
        return value

    return function_class(kernel)


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
    if "asc._C.libpyasc" in sys.modules or any((root / "_C").glob("libpyasc*.so")):
        # CI with the compiled extension uses the regular package imports.
        from asc.codegen.function import Function
        return Function
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


def test_unreferenced_global_callable_does_not_reject_legal_closure(function_class):
    kernel = make_kernel(7, function_class)
    kernel.cache_key
    captured = [val for (name, _), (val, _) in kernel.used_global_vals.items() if name == "value"]
    assert len(captured) == 1
    assert captured[0] == kernel.compute_globals()["value"] == 7


def test_dependency_value_matches_codegen_closure_with_same_named_global(function_class, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "value", -1)
    kernel = make_kernel(9, function_class)
    kernel.cache_key
    captured = [val for (name, _), (val, _) in kernel.used_global_vals.items() if name == "value"]
    assert captured == [kernel.compute_globals()["value"]] == [9]


def test_nonlocal_none_does_not_fall_back_to_same_named_global(function_class):
    kernel = make_kernel(None, function_class)
    kernel.cache_key
    assert not any(name == "value" for name, _ in kernel.used_global_vals)


def global_kernel():
    return value


def test_global_only_reference_still_records_the_global(function_class, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "value", -1)
    kernel = function_class(global_kernel)
    kernel.cache_key
    captured = [val for (name, _), (val, _) in kernel.used_global_vals.items() if name == "value"]
    assert len(captured) == 1
    assert captured[0] == -1
