# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from unittest.mock import patch

import pytest

import asc
from asc.runtime.jit import MockTensor


@pytest.fixture(autouse=True)
def mock_jit():
    with patch("asc.runtime.jit.JITFunction._run_launcher", return_value=None), \
    patch("asc.runtime.jit.JITFunction._run_compiler", return_value=None):
        yield


@asc.jit
def func_visit_list(a, b, c):
    nums = [a, b, c]
    return nums


def test_func_visit_list(filecheck):

    @asc.jit
    def func_visit_list_kernel():
        # CHECK: return %arg0, %arg1, %arg2 : i32, i32, i32
        func_visit_list(1, 2, 3)

    filecheck(func_visit_list_kernel)()


def test_func_visit_bool_op(filecheck):

    @asc.jit
    def func_visit_bool_op_kernel(value, min_threshold, max_threshold, cnt, step):
        # CHECK: arith.andi
        # CHECK: arith.ori
        if value >= min_threshold and value <= max_threshold:
            cnt += step
        elif value < min_threshold or value > max_threshold:
            cnt += step * 2
        ret = cnt == step

    filecheck(func_visit_bool_op_kernel)(10, 5, 20, 0, 1)


def test_func_visit_compare(filecheck):

    @asc.jit
    def func_visit_compare_kernel(a, b, c):
        # CHECK: arith.cmpi
        ans = 0
        if a > b:
            ans += 1
        if a + b < c:
            ans += 1
        if ans >= 10:
            ans += 1
        if ans <= 5:
            ans += 1
        if ans == c:
            ans += 1
        ret = ans

    filecheck(func_visit_compare_kernel)(2, 1, 2)


@asc.jit
def func_visit_constant():
    a = 1
    b = True
    return a, b


def test_func_visit_constant(filecheck):

    @asc.jit
    def func_visit_constant_kernel():
        # CHECK: %c1_i32 = arith.constant 1
        # CHECK: %true = arith.constant true
        func_visit_constant()

    filecheck(func_visit_constant_kernel)()


def test_func_visit_if_exp(filecheck):

    @asc.jit
    def func_visit_if_exp_kernel(x):
        # CHECK: scf.if %0 -> (i32) {
        # CHECK: } else {
        ans = x if x > 0 else 0
        ret = ans

    filecheck(func_visit_if_exp_kernel)(10)


def test_func_visit_if(filecheck):

    @asc.jit
    def func_visit_if_kernel(x, y, z, ans, step):
        # CHECK: scf.if %1 -> (i32) {
        # CHECK: } else {
        # CHECK: scf.if %5 -> (i32) {
        # CHECK: } else {
        if x + y == z:
            ans += step
        elif x + y > z:
            ans += step
        else:
            ans += step
        ret = ans == 1

    filecheck(func_visit_if_kernel)(1, 2, 5, 0, 1)


def test_func_visit_pass(filecheck):

    @asc.jit
    def func_visit_pass():
        # CHECK: func.func @func_visit_pass()
        # CHECK-NEXT: return
        pass

    filecheck(func_visit_pass)()


@asc.jit
def nested_void_leaf(value):
    result = value + 1


@asc.jit
def nested_void_middle(value):
    for _ in asc.range(1):
        nested_void_leaf(value)


def test_nested_void_jit_functions(filecheck):

    @asc.jit
    def nested_void_kernel(value):
        # CHECK-DAG: func.func @nested_void_middle
        # CHECK-DAG: call @nested_void_leaf
        # CHECK-DAG: call @nested_void_middle
        nested_void_middle(value)

    filecheck(nested_void_kernel)(1)


@asc.jit
def func_visit_return(x):
    return x


def test_func_visit_return(filecheck):

    @asc.jit
    def func_visit_return_kernel():
        # CHECK: return %arg0 : i32
        func_visit_return(100)

    filecheck(func_visit_return_kernel)()


@asc.jit
def func_visit_tuple(x, y, z):
    return x, y, z


def test_func_visit_tuple(filecheck):

    @asc.jit
    def func_visit_tuple_kernel():
        # CHECK: return %arg0, %arg1, %arg2 : i32, i32, i32
        func_visit_tuple(1, 2, 3)

    filecheck(func_visit_tuple_kernel)()


def test_func_visit_unary_op(filecheck):

    @asc.jit
    def func_visit_unary_op_kernel(a):
        # CHECK: arith.constant -2
        ret = a + ~1

    filecheck(func_visit_unary_op_kernel)(1)


def test_func_visit_while(filecheck):

    @asc.jit
    def func_visit_while_kernel(n):
        # CHECK: scf.while
        ans = 0
        i = 0
        while i < n:
            ans += i
            i += 1
        ret = ans

    filecheck(func_visit_while_kernel)(10)


def test_func_visit_augassign(filecheck):

    @asc.jit
    def func_visit_augassign_kernel(a, b):
        # CHECK: arith.addi
        # CHECK: arith.muli
        # CHECK: arith.subi
        # CHECK: arith.divsi
        # CHECK: arith.remsi
        result = a
        result += b
        result *= 2
        result -= 5
        result /= a
        result %= 3
        ret = result

    filecheck(func_visit_augassign_kernel)(2, 5)


def test_func_visit_ann_assign(filecheck):

    @asc.jit
    def func_visit_ann_assign_kernel(num1, num2, num3):
        # CHECK: %c1_i32 = arith.constant 1
        # CHECK: %c3_i32 = arith.constant 3
        count: int = 3
        sum_result: float = 1.0
        sum_result += num1
        sum_result += num2
        sum_result += num3
        result = sum_result / count

    filecheck(func_visit_ann_assign_kernel)(12, 18, 100)


def test_func_visit_arguments(filecheck):

    @asc.jit
    def func_visit_arguments(data: asc.GlobalAddress, threshold: int, flag: bool) -> None:
        # CHECK: %arg0: memref<*xi32, 22>
        # CHECK: %arg1: i32
        # CHECK: %arg2: i8
        pass

    data = MockTensor(asc.int32)
    filecheck(func_visit_arguments)(data, 32, True)


def test_error_test_str():

    @asc.jit
    def func_error_test_str(name):
        return name

    with pytest.raises(TypeError) as e:
        func_error_test_str[1]("test")
    assert "Argument type in JIT function is not supported: str" in str(e.value)


def test_builtin_print_dispatch(filecheck):

    @asc.jit
    def func_builtin_print(x):
        # CHECK: ascendc.printf
        print(x)

    filecheck(func_builtin_print)(100)


def test_builtin_assert_dispatch(filecheck):

    @asc.jit
    def func_builtin_assert(x, num: asc.ConstExpr):
        # CHECK: scf.if
        # CHECK: ascendc.printf
        # CHECK: ascendc.trap
        assert x >= 0, f"assert failed {num}"

    filecheck(func_builtin_assert)(0, 100)


def test_func_visit_floordiv_int(filecheck):

    @asc.jit
    def func_visit_floordiv_int_kernel(a, b):
        # CHECK: arith.divsi
        # CHECK: arith.remsi
        # CHECK: arith.cmpi ne
        # CHECK: arith.cmpi slt
        # CHECK: arith.cmpi slt
        # CHECK: arith.xori
        # CHECK: arith.andi
        # CHECK: arith.select
        # CHECK: arith.subi
        result = a // b

    filecheck(func_visit_floordiv_int_kernel)(-5, 2)


def test_func_visit_rfloordiv_int(filecheck):

    @asc.jit
    def func_visit_rfloordiv_int_kernel(b):
        # CHECK: arith.divsi
        # CHECK: arith.remsi
        # CHECK: arith.select
        result = -5 // b

    filecheck(func_visit_rfloordiv_int_kernel)(2)


def test_func_visit_floordiv_float(filecheck):

    @asc.jit
    def func_visit_floordiv_float_kernel(a, b):
        # CHECK: arith.divf
        # CHECK: math.floor
        result = a // b

    filecheck(func_visit_floordiv_float_kernel)(5.0, 2.0)


def test_func_visit_rfloordiv_float(filecheck):

    @asc.jit
    def func_visit_rfloordiv_float_kernel(b):
        # CHECK: arith.divf
        # CHECK: math.floor
        result = -5.0 // b

    filecheck(func_visit_rfloordiv_float_kernel)(2.0)


def test_func_visit_truediv_control(filecheck):

    @asc.jit
    def func_visit_truediv_control_kernel(a, b):
        # CHECK: arith.divf
        # CHECK-NOT: math.floor
        result = a / b

    filecheck(func_visit_truediv_control_kernel)(5.0, 2.0)