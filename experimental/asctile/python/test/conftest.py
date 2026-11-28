# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from contextlib import ExitStack
import importlib
import importlib.util
import os
import types
from unittest.mock import patch

from asc.experimental.asctile.interpreter import interpreter_enabled
from asc.lib.profiling import Profiler, task_time_median
from asc.runtime import config
import pytest


class StubProfiler:

    def profile(self, *args, **kwargs):
        return self

    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--backend", type=config.Backend, default=config.Backend.Model, help="Runtime backend")
    parser.addoption("--platform", type=config.Platform, default=config.Platform.Ascend950PR_9599,
                     help="Runtime platform")
    parser.addoption("--device", type=int, help="Device ID")
    parser.addoption("--profile", action="store_true", help="Enable NPU profiling (if available)")
    parser.addoption("--profile-path", type=str, default=None, help="Directory to save profiling results")
    parser.addoption("--runs", type=int, default=1, help="Number of kernel launches")
    parser.addoption("--compile-only", action="store_true", help="Stop after the kernel compilation (do not launch)")
    parser.addoption("--torch-seed", type=int, default=0, help="Define the seed value for torch.manual_seed(...)")


def pytest_configure(config):
    config.profiling_results = []
    config.addinivalue_line(
        "markers",
        "asctile_xfail(reason, compile_ok, when): expected failure; the apply_xfail fixture attaches xfail at run time",
    )
    config.addinivalue_line("markers", "no_interpret: the test is not supported by the interpreter mode")


def pytest_make_parametrize_id(val):
    if callable(val):
        return val.__name__
    return str(val)


def pytest_terminal_summary(terminalreporter, config):
    if not config.profiling_results:
        return
    terminalreporter.write_sep("=", "Profiling results")
    for entry in config.profiling_results:
        terminalreporter.write_line(f"{entry['test']}: {entry['duration']} μs")


@pytest.fixture(autouse=True)
def skip_no_interpret(request: pytest.FixtureRequest):
    if request.node.get_closest_marker("no_interpret") and interpreter_enabled():
        pytest.skip("the test is not supported by the interpreter mode")


@pytest.fixture(autouse=True)
def apply_xfail(request: pytest.FixtureRequest, compile_only):
    mark = request.node.get_closest_marker("asctile_xfail")
    if mark is None:
        return
    reason = mark.kwargs.get("reason", "")
    compile_ok = mark.kwargs.get("compile_ok", False)
    when = mark.kwargs.get("when")
    if compile_ok and compile_only:
        return
    if when is not None:
        callspec = getattr(request.node, "callspec", None)
        ctx = types.SimpleNamespace(
            params=callspec.params if callspec else {},
            config=request.config,
            node=request.node,
        )
        if not when(ctx):
            return
    request.node.add_marker(pytest.mark.xfail(reason=reason, strict=True))


@pytest.fixture
def backend(request: pytest.FixtureRequest):
    return request.config.getoption("--backend")


@pytest.fixture
def platform(request: pytest.FixtureRequest):
    return request.config.getoption("--platform")


@pytest.fixture
def device_id(request: pytest.FixtureRequest):
    return request.config.getoption("--device", default=None)


@pytest.fixture(autouse=True)
def set_platform(request: pytest.FixtureRequest, backend: config.Backend, platform: config.Platform, device_id: int):
    if not request.node.get_closest_marker("no_set_platform"):
        config.set_platform(backend, platform, device_id, check=False)
    yield


@pytest.fixture
def require_c310(platform: config.Platform):

    def require_c310_impl():
        if config.platform_to_arch(platform) != config.CompilationArch.C310:
            pytest.skip(f"{platform.value} platform is not supported")

    return require_c310_impl


@pytest.fixture
def profiler(request, tmp_path_factory, backend):
    if backend != config.Backend.NPU or not request.config.getoption("--profile"):
        yield StubProfiler()
        return
    profiling_path = request.config.getoption("--profile-path")
    if profiling_path is not None:
        safe_name = request.node.nodeid.replace(":", "_").replace("/", "_").replace(" ", "")
        result_path = os.path.join(profiling_path, safe_name)
        os.makedirs(result_path, exist_ok=True)
    else:
        result_path = str(tmp_path_factory.mktemp("profiler"))
    profiler = Profiler(result_path)
    yield profiler
    request.config.profiling_results.append({
        "test": request.node.nodeid,
        "duration": task_time_median(profiler.last_result.tasks, skip=1),
    })


@pytest.fixture
def runs(request: pytest.FixtureRequest):
    return request.config.getoption("--runs")


@pytest.fixture(scope="session", autouse=True)
def compile_only(request: pytest.FixtureRequest):
    if not request.config.getoption("--compile-only"):
        yield False
        return
    os.environ["PYASC_DRY_RUN"] = "1"
    with ExitStack() as stack:
        try:
            importlib.import_module("numpy")
            stack.enter_context(patch("numpy.testing.assert_allclose"))
        except ModuleNotFoundError:
            pass
        try:
            importlib.import_module("torch")
            stack.enter_context(patch("torch.testing.assert_close"))
            for fn in ("torch.allclose", "torch.equal"):
                stack.enter_context(patch(fn)).return_value = True
        except ModuleNotFoundError:
            pass
        yield True


@pytest.fixture
def torch_seed(request: pytest.FixtureRequest):
    return request.config.getoption("--torch-seed")


@pytest.fixture(autouse=True)
def configure_torch(backend: config.Backend, device_id: int, torch_seed: int):
    try:
        import torch
        if backend == config.Backend.NPU and importlib.util.find_spec("torch_npu") is not None:
            device = "npu" if device_id is None else f"npu:{device_id}"
        else:
            device = "cpu"
        with torch.device(device):
            torch.manual_seed(torch_seed)
            yield
    except ModuleNotFoundError:
        yield
