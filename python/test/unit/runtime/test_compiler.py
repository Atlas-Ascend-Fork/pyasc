# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch, MagicMock

from asc.lib.runtime import CoreType
from asc.runtime.compiler import CompilationArch, CompileOptions, Compiler
from asc.runtime import config
import pytest


@pytest.fixture(autouse=True)
def disable_dump(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PYASC_DUMP_PATH", raising=False)
    monkeypatch.setenv("ASCENDC_DUMP", "False")
    yield


@contextmanager
def patch_passes():
    pass_functions = [
        "asc._C.passes.ascendc.add_declare_py_struct",
        "asc._C.passes.ascendc.add_detect_enable_debug",
        "asc._C.passes.ascendc.add_detect_kernel_type",
        "asc._C.passes.ascendc.add_generate_boilerplate",
        "asc._C.passes.ascendc.add_legalize_kernel_args",
        "asc._C.passes.ascendc.add_privatize_func",
        "asc._C.passes.ascendc.add_erase_sync",
        "asc._C.passes.ascendc.add_hoist_que_bind",
        "asc._C.passes.ascendc.add_hoist_tensor_allocation",
        "asc._C.passes.ascendc.add_input_output_tensor",
        "asc._C.passes.ascendc.add_insert_que_sync",
        "asc._C.passes.ascendc.add_materialize_tensor",
        "asc._C.passes.ascendc.add_unify_pipe",
        "asc._C.passes.ascendc.add_verify_sync",
        "asc._C.passes.common.add_canonicalizer",
        "asc._C.passes.common.add_cse",
        "asc._C.passes.common.add_inliner",
        "asc._C.passes.common.add_licm",
        "asc._C.passes.common.add_reconcile_unrealized_casts",
        "asc._C.passes.common.add_sccp",
        "asc._C.passes.common.add_strip_debug_info",
        "asc._C.passes.common.add_symbol_dce",
    ]
    patches = [patch(name, return_value=None) for name in pass_functions]
    try:
        mocks = []
        for patch_obj in patches:
            mocks.append(patch_obj.start())
        yield mocks
    finally:
        for patch_obj in patches:
            patch_obj.stop()


@pytest.fixture
def mock_popen():
    mock_pass_manager = MagicMock()
    mock_pass_manager.enable_verifier.return_value = None
    mock_pass_manager.enable_printing.return_value = None
    mock_pass_manager.run.return_value = None
    mock_process = MagicMock()
    mock_process.communicate.return_value = ("output", 0)
    mock_process.returncode = 0
    with \
    patch("asc._C.passes.PassManager", return_value=mock_pass_manager), \
    patch_passes(), \
    patch("asc._C.ir.Builder"), \
    patch("asc._C.ir.get_kernel_arg_attrs", return_value=None), \
    patch("asc._C.translation.ir_to_ascendc", return_value="mock_translation"), \
    patch("pathlib.Path.read_bytes", return_value=None), \
    patch("shutil.which", return_value="_bisheng_"), \
    patch("subprocess.Popen", return_value=mock_process) as mock:
        yield mock


@pytest.fixture
def mock_ir_module():
    mod = MagicMock()
    mod.get_context.return_value = None
    yield mod


def test_invalid_compile_option():
    with pytest.raises(RuntimeError, match="Please check input compile option"):
        invalid_options = CompileOptions(opt_level=0)
        Compiler(invalid_options)


@pytest.mark.parametrize("log_path, expected_option", [(None, None), ("", None),
                                                       ("sync.log", "-cce-auto-sync-log=sync.log")])
def test_auto_sync_log_is_only_passed_for_a_nonempty_path(log_path, expected_option):
    compiler = object.__new__(Compiler)
    compiler.options = CompileOptions(auto_sync_log=log_path)
    compiler.arch = CompilationArch.C220
    compiler.compiler = "bisheng"
    compiler.enable_debug = False

    command = compiler._get_compiler_cmd("dav-c220-vec", Path("kernel.cce"), Path("kernel.o"), [])
    log_options = [option for option in command if option.startswith("-cce-auto-sync-log=")]

    assert log_options == ([] if expected_option is None else [expected_option])


def test_kernel_type_mix(mock_ir_module, mock_popen):
    options = CompileOptions(kernel_type=config.KernelType.MIX_AIC_1_1)
    compiler = Compiler(options)
    ret = compiler.run(mock_ir_module, "test_func")
    assert mock_popen.call_count == 3
    assert ret.meta.core_type == CoreType.AiCore


def test_kernel_type_hard_sync(mock_ir_module, mock_popen):
    options = CompileOptions(kernel_type=config.KernelType.MIX_AIV_HARD_SYNC)
    compiler = Compiler(options)
    ret = compiler.run(mock_ir_module, "test_func")
    assert mock_popen.call_count == 2
    assert ret.meta.core_type == CoreType.VectorCore


def test_kernel_type_aic_only(mock_ir_module, mock_popen):
    options = CompileOptions(kernel_type=config.KernelType.AIC_ONLY)
    compiler = Compiler(options)
    ret = compiler.run(mock_ir_module, "test_func")
    assert mock_popen.call_count == 2
    assert ret.meta.core_type == CoreType.CubeCore


def test_kernel_type_cmd_exec_failed(mock_ir_module, mock_popen):
    mock_popen.return_value.configure_mock(**{"communicate.return_value": (b"output", 0), "returncode": 1})
    options = CompileOptions(kernel_type=config.KernelType.MIX_AIV_HARD_SYNC)
    compiler = Compiler(options)
    with pytest.raises(RuntimeError, match="Please rerun"):
        compiler.run(mock_ir_module, "test_func")
    # 2 calls: initial compile + one "-disable-machine-licm" retry; the retry verdict
    # checks proc.returncode (not the stderr slot), so a failed retry raises immediately.
    assert mock_popen.call_count == 2


def test_dry_run_requested_treats_zero_as_off():
    from asc.runtime.launcher import dry_run_requested

    assert dry_run_requested(None) is False
    assert dry_run_requested("") is False
    assert dry_run_requested("0") is False
    assert dry_run_requested("false") is False
    assert dry_run_requested("1") is True
    assert dry_run_requested("yes") is True
