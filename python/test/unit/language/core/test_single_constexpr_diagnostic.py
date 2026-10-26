# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check constexpr diagnostics through the real asc package import."""
from typing import Union

import pytest

from asc.language.core.constexpr import ConstExpr, require_constexpr


@pytest.mark.parametrize("constraint,value,expected", [(int, "wrong", "int"), (str, 3, "str"), (int, None, "int"),
                                                       (int, 1.2, "int")])
def test_single_required_type_is_visible_in_error(constraint, value, expected):
    with pytest.raises(RuntimeError) as error:
        require_constexpr(value, constraint, arg_name="depth")
    assert f"ConstExpr[{expected}] is required for 'depth' argument" in str(error.value)
    assert value.__class__.__name__ in str(error.value)


def test_duplicate_constraints_do_not_hide_the_required_type():
    with pytest.raises(RuntimeError, match=r"ConstExpr\[int\] is required"):
        require_constexpr("wrong", int, int)


def test_wrapped_bad_value_preserves_the_same_type_and_argument_name():
    wrapped = ConstExpr("wrong")
    with pytest.raises(RuntimeError, match=r"ConstExpr\[int\] is required for 'pos' argument"):
        require_constexpr(wrapped, int, arg_name="pos")


@pytest.mark.parametrize("value,constraints", [(3, (int, )), (True, (int, )), ("valid", (str, )), (1.5, (int, float)),
                                               (2, (Union[int, float], ))])
def test_valid_controls_are_still_accepted(value, constraints):
    assert require_constexpr(value, *constraints) is None
    assert require_constexpr(ConstExpr(value), *constraints) is None


def test_multiple_type_error_keeps_existing_union_text():
    with pytest.raises(RuntimeError, match=r"ConstExpr\[int \| float\] is required"):
        require_constexpr("wrong", int, float)


def test_no_constraint_control_still_raises():
    with pytest.raises(RuntimeError, match="Constraints must be provided"):
        require_constexpr(3)
