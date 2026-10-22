# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run complete source hashing on real file trees; never load a native SDK."""
import importlib.util
from pathlib import Path
import sys
import sysconfig

import pytest


def load_cache(source):
    name = "pyasc_runtime_hash_" + str(id(source))
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hash_tree(tmp_path):
    actual_runtime = Path(__file__).resolve().parents[3] / "asc/runtime"
    package = tmp_path / "asc"
    runtime = package / "runtime"
    runtime.mkdir(parents=True)
    for filename in ("cache.py", "compiler.py", "launcher.py", "jit.py"):
        (runtime / filename).write_bytes((actual_runtime / filename).read_bytes())
    for component in ("codegen", "language"):
        directory = package / component
        directory.mkdir()
        (directory / "hash_input.py").write_text("VALUE = 1\n")
    binary = package / "_C" / ("libpyasc" + sysconfig.get_config_var("EXT_SUFFIX"))
    binary.parent.mkdir()
    # The production hash only reads bytes. This deliberately non-executable
    # checksum fixture is never imported or used as a native SDK replacement.
    binary.write_bytes(b"checksum input; never loaded or executed")
    return package, load_cache(runtime / "cache.py"), binary


def changed_key(module, path):
    first = module.pyasc_key()
    path.write_bytes(path.read_bytes() + b"\n# source change\n")
    # Recompute as a new process after a package update would do.
    module.pyasc_key.cache_clear()
    return first, module.pyasc_key()


@pytest.mark.parametrize("filename", ["compiler.py", "launcher.py", "jit.py"])
def test_runtime_source_changes_invalidate_key(hash_tree, filename):
    package, module, _ = hash_tree
    old, new = changed_key(module, package / "runtime" / filename)
    assert old != new


@pytest.mark.parametrize("component", ["codegen", "language"])
def test_existing_frontend_source_invalidation_is_preserved(hash_tree, component):
    package, module, _ = hash_tree
    old, new = changed_key(module, package / component / "hash_input.py")
    assert old != new


def test_existing_backend_byte_invalidation_is_preserved(hash_tree):
    _, module, binary = hash_tree
    old, new = changed_key(module, binary)
    assert old != new


def test_unrelated_file_does_not_change_key(hash_tree):
    package, module, _ = hash_tree
    old = module.pyasc_key()
    (package / "notes.txt").write_text("unrelated documentation\n")
    module.pyasc_key.cache_clear()
    assert old == module.pyasc_key()
