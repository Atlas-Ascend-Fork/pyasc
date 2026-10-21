# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run the complete production command helper against real CPU subprocesses."""
import ast
from pathlib import Path
import subprocess
import sys

import pytest


def load_run_cmd():
    source = Path(__file__).resolve().parents[3] / "asc/runtime/compiler.py"
    tree = ast.parse(source.read_text())
    compiler = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Compiler")
    helper = next(node for node in compiler.body if isinstance(node, ast.FunctionDef) and node.name == "_run_cmd")
    # This static helper only needs subprocess. Preserve its entire production
    # body while avoiding imports that initialize the unrelated native SDK.
    helper.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), helper],
        type_ignores=[])
    namespace = {"subprocess": subprocess}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    function = namespace.get("_run_cmd")
    assert callable(function)
    return function


@pytest.mark.parametrize("cmd_type", ["compile", "link"])
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_non_utf8_failure_preserves_command_diagnostic(cmd_type, stream):
    helper = load_run_cmd()
    code = f"import sys; sys.{stream}.buffer.write(bytes([255])+b' native diagnostic '+bytes([254])); sys.exit(7)"
    command = [sys.executable, "-c", code]
    with pytest.raises(RuntimeError) as caught:
        helper(command, cmd_type)
    message = str(caught.value)
    assert cmd_type + " failed!" in message
    assert "native diagnostic" in message
    assert "\ufffd" in message
    assert "Please rerun " + " ".join(command) in message


@pytest.mark.parametrize("cmd_type", ["compile", "link"])
def test_utf8_failure_preserves_valid_text(cmd_type):
    helper = load_run_cmd()
    command = [sys.executable, "-c", "import sys; print('valid diagnostic'); sys.exit(7)"]
    with pytest.raises(RuntimeError, match="valid diagnostic"):
        helper(command, cmd_type)


@pytest.mark.parametrize("cmd_type", ["compile", "link"])
def test_success_does_not_decode_output(cmd_type):
    helper = load_run_cmd()
    command = [sys.executable, "-c", "import sys; sys.stdout.buffer.write(bytes([255]))"]
    assert helper(command, cmd_type) is None


def test_successful_existing_compile_retry_is_preserved():
    helper = load_run_cmd()
    code = ("import sys; sys.stdout.buffer.write(bytes([255])); "
            "sys.exit(0 if '-disable-machine-licm' in sys.argv else 7)")
    command = [sys.executable, "-c", code, "--cce-aicore-only"]
    assert helper(command, "compile") is None
    assert "-disable-machine-licm" in command


def test_failed_existing_compile_retry_preserves_diagnostic():
    helper = load_run_cmd()
    code = "import sys; sys.stdout.buffer.write(bytes([255])+b' retry diagnostic'); sys.exit(7)"
    command = [sys.executable, "-c", code, "--cce-aicore-only"]
    with pytest.raises(RuntimeError, match="retry diagnostic"):
        helper(command, "compile")
    assert "-disable-machine-licm" in command
