# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Indexed JIT callables must capture independent launch options."""
from dataclasses import dataclass
from types import SimpleNamespace

from asc.runtime import jit as jit_module
from asc.runtime.launcher import LaunchOptions


@dataclass
class _Options:
    marker: int = 0


def _make_jit(launched_options):

    class IndexedJIT:
        """Execute the real indexing and run methods against stubbed codegen/compiler/launcher."""

        __getitem__ = jit_module.JITFunction.__getitem__
        _run = jit_module.JITFunction._run
        if hasattr(jit_module.JITFunction, "_run_with_options"):
            _run_with_options = jit_module.JITFunction._run_with_options
        extract_kwargs = staticmethod(jit_module.JITFunction.extract_kwargs)
        get_arg_type = staticmethod(jit_module.JITFunction.get_arg_type)
        split_args = staticmethod(jit_module.Function.split_args)

        def __init__(self, launched):

            def kernel():
                return None

            self.fn = kernel
            self.fn_name = "test_kernel"
            self.default_options = {}
            self.launch_options = LaunchOptions()
            self.kernel_cache = {}
            self.codegen = SimpleNamespace(options_cls=_Options)
            self.compiler = SimpleNamespace(options_cls=_Options)
            self.launcher = SimpleNamespace(options_cls=LaunchOptions)
            self._compile_and_cache = lambda prereqs: (object(), "cache_key")
            self._run_launcher = lambda kernel, options, runtime_args, mem_cache_key:\
                launched.append(options)

    return IndexedJIT(launched_options)


def test_indexed_callables_use_captured_launch_options():
    launched = []
    jit = _make_jit(launched)

    one_core = jit[1]
    two_cores = jit[2]
    one_core()
    two_cores()

    assert [options.core_num for options in launched] == [1, 2]


def test_indexed_callable_with_stream_uses_captured_stream():
    stream_one = object()
    stream_two = object()
    launched = []
    jit = _make_jit(launched)

    on_stream_one = jit[1, stream_one]
    on_stream_two = jit[2, stream_two]
    on_stream_one()
    on_stream_two()

    assert [(options.core_num, options.stream) for options in launched] == [(1, stream_one), (2, stream_two)]


def test_direct_run_keeps_default_launch_options():
    launched = []
    jit = _make_jit(launched)

    jit._run()

    assert launched == [jit.launch_options]
