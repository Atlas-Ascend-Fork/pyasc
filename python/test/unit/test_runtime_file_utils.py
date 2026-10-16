# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import shutil
import stat
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def file_utils_cls():
    module_path = Path(__file__).resolve().parents[2] / "asc" / "runtime" / "utils.py"
    spec = importlib.util.spec_from_file_location("asc_runtime_file_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module.FileUtils


@pytest.fixture
def relative_dump_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    unique = "pyasc_dump_" + uuid4().hex
    relative = Path("tmp") / unique / "nested"
    yield relative
    misplaced = Path("/tmp") / unique
    if misplaced.exists():
        shutil.rmtree(misplaced)


@pytest.mark.parametrize("mode", [0o750, 0o700])
def test_create_dir_relative_path(file_utils_cls, relative_dump_path, mode):
    file_utils_cls.create_dir(relative_dump_path, mode)
    assert relative_dump_path.is_dir()
    assert stat.S_IMODE(relative_dump_path.stat().st_mode) == mode
    assert stat.S_IMODE(relative_dump_path.parent.stat().st_mode) == mode


def test_dump_file_after_relative_directory_creation(file_utils_cls, relative_dump_path):
    file_utils_cls.create_dir(relative_dump_path)
    file_utils_cls.dump_file(relative_dump_path, "trace.txt", "new diagnostic")
    assert (relative_dump_path / "trace.txt").read_text() == "new diagnostic"


@pytest.mark.parametrize("mode", [0o750, 0o700])
def test_create_dir_absolute_path(file_utils_cls, tmp_path, mode):
    target = tmp_path / "parent" / "child"
    file_utils_cls.create_dir(target, mode)
    assert target.is_dir()
    assert stat.S_IMODE(target.stat().st_mode) == mode
    assert stat.S_IMODE(target.parent.stat().st_mode) == mode


def test_create_dir_none(file_utils_cls):
    assert file_utils_cls.create_dir(None) is None


def test_existing_relative_directory_retains_mode(file_utils_cls, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = Path("existing")
    target.mkdir(mode=0o700)
    file_utils_cls.create_dir(target, 0o750)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700


@pytest.mark.parametrize("data,expected", [
    ("diagnostic", b"diagnostic"),
    (b"binary diagnostic", b"binary diagnostic"),
    (lambda: "generated", b"generated"),
])
def test_dump_file_absolute_control(file_utils_cls, tmp_path, data, expected):
    target = tmp_path / "dump"
    file_utils_cls.create_dir(target)
    file_utils_cls.dump_file(target, "trace", data)
    assert (target / "trace").read_bytes() == expected
