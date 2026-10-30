# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exercise overload resolution through the normally installed asc package."""

import textwrap

import pytest

from asc.language.core.utils import OverloadDispatcher


def compiled_function(source):
    namespace = {}
    exec(compile(textwrap.dedent(source), "host-overload-example", "exec", dont_inherit=True), namespace)
    function = namespace.get("sample")
    assert callable(function)
    return function


@pytest.mark.parametrize("source", [
    """
    from __future__ import annotations
    def sample(value: int, scale: int = 2) -> int:
        return value * scale
    """,
    """
    def sample(value: "int", scale: "int" = 2):
        return value * scale
    """,
    """
    class Item:
        def __init__(self, value):
            self.value = value
    def sample(value: "Item"):
        return value.value
    """,
])
def test_resolved_string_annotations_dispatch_actual_python_function(source):
    sample = compiled_function(source)
    dispatcher = OverloadDispatcher("sample")
    dispatcher.register_auto(sample)
    item = sample.__globals__.get("Item")
    if item is None:
        assert dispatcher(3) == 6
        assert dispatcher(value=4, scale=3) == 12
    else:
        assert dispatcher(item(9)) == 9


def test_eager_annotations_and_defaults_keep_working():
    sample = compiled_function("""
        def sample(value: int, scale: int = 2):
            return value * scale
    """)
    dispatcher = OverloadDispatcher("sample")
    dispatcher.register_auto(sample)
    assert dispatcher(3) == 6
    assert dispatcher(4, scale=3) == 12


def test_missing_parameter_annotation_keeps_clear_registration_error():
    sample = compiled_function("""
        def sample(value):
            return value
    """)
    with pytest.raises(ValueError, match="does not have a type annotation"):
        OverloadDispatcher("sample").register_auto(sample)


def test_incompatible_eager_argument_keeps_dispatch_diagnostic():
    sample = compiled_function("""
        def sample(value: int):
            return value
    """)
    dispatcher = OverloadDispatcher("sample")
    dispatcher.register_auto(sample)
    with pytest.raises(RuntimeError, match="No viable candidates"):
        dispatcher("wrong type")
