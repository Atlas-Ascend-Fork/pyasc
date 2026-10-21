# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Execute the complete real support module and use actual ctypes storage."""
import ctypes
import importlib.util
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "python/asc/lib/runtime/support.py"
NAMES = [
    "l0a_size", "l0b_size", "l0c_size", "l1_size", "ub_size", "l2_size", "l2_page_num", "block_size", "bank_size",
    "bank_num", "burst_in_one_block", "bank_group_num"
]
VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 2**40 + 9, 2**40 + 10, 2**40 + 11, 2**40 + 12]


@pytest.fixture
def support(monkeypatch):
    spec = importlib.util.spec_from_file_location("host_only_pyasc_support", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_declared_handle_owns_64_bytes_of_storage(support):
    assert ctypes.sizeof(support.MemorySizes.Handle) == 64
    assert ctypes.alignment(support.MemorySizes.Handle) == 8


def test_declared_fields_have_native_offsets(support):
    offsets = [getattr(support.MemorySizes.Handle, name).offset for name in NAMES]
    assert offsets == [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56]


def test_constructed_fields_serialize_to_real_native_bytes(support):
    handle = support.MemorySizes.Handle(**dict(zip(NAMES, VALUES)))
    assert bytes(handle) == struct.pack("@8I4Q", *VALUES)


def test_buffer_roundtrip_restores_all_fields(support):
    expected = struct.pack("@8I4Q", *VALUES)
    handle = support.MemorySizes.Handle.from_buffer_copy(expected)
    assert [getattr(handle, name) for name in NAMES] == VALUES
    handle.bank_size = 2**48 + 3
    assert struct.unpack("@8I4Q", bytes(handle))[8] == 2**48 + 3


def test_dataclass_values_are_unchanged(support):
    sizes = support.MemorySizes(**dict(zip(NAMES, VALUES)))
    assert [getattr(sizes, name) for name in NAMES] == VALUES
    assert sizes == support.MemorySizes(**dict(zip(NAMES, VALUES)))


def test_existing_runtime_structure_controls_keep_storage(support):
    assert ctypes.sizeof(support.DevBinary) > 0
    binary = support.DevBinary(magic=3, version=2, data=b"abc", length=3)
    assert binary.magic == 3 and binary.length == 3
    assert ctypes.sizeof(support.ProfilingCommandHandle) > 0
    profiling = support.ProfilingCommandHandle(profSwitch=123)
    assert profiling.profSwitch == 123
