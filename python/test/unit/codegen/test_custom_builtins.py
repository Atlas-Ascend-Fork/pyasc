# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Named builtin cache identities stay stable across order and processes."""

import subprocess
import sys

from asc.codegen.function_visitor import CustomBuiltins


def first_builtin(value):
    return value


def second_builtin(value):
    return value + 1


def test_repr_carries_named_builtin_identity():
    first = CustomBuiltins(convert=first_builtin)
    second = CustomBuiltins(convert=second_builtin)
    assert repr(first) != repr(second)
    assert str(first) == repr(first)


def test_mapping_order_does_not_change_repr():
    assert repr(CustomBuiltins(a=first_builtin,
                               b=second_builtin)) == repr(CustomBuiltins(b=second_builtin, a=first_builtin))


def test_repr_uses_module_and_qualname():

    def custom_assert(value):
        return value

    custom_assert.__module__ = "example.runtime.custom_builtins"
    custom_assert.__qualname__ = "custom_assert"
    assert repr(CustomBuiltins({"assert": custom_assert
                                })) == ("CustomBuiltins(assert=example.runtime.custom_builtins.custom_assert)")


def test_default_mainline_builtin_identity_stays_stable_across_processes():
    code = ("from asc.codegen.function_visitor import CustomBuiltins; "
            "from asc.language.basic.debug import device_assert, device_print; "
            "from asc.language.core.range import range; "
            "print(repr(CustomBuiltins({'assert': device_assert, 'print': device_print, 'range': range})))")
    assert subprocess.check_output([sys.executable, '-c',
                                    code]) == subprocess.check_output([sys.executable, '-c', code])
