# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
from typing import Dict, NoReturn, Type

from asc.codegen.function_visitor import FunctionVisitor as FunctionVisitorBase
from asc.language.core.ir_value import IRHandle, IRValue, materialize_ir_value

from ..language.context_manager import ContextManager


class FunctionVisitor(FunctionVisitorBase):

    def visit_While(self, node: ast.While) -> NoReturn:
        self.raise_unsupported(node, "'while' statement is not supported in JIT function")

    def visit_With(self, node: ast.With) -> None:
        if len(node.items) != 1:
            self.raise_unsupported(node, "Only one item in with-statement is supported")
        item = node.items[0]
        context = self.visit(item.context_expr)
        with self.nest_scope():
            entered = context.__enter__()
            if isinstance(item.optional_vars, ast.Name):
                self.scope.save(self.visit(item.optional_vars), entered)
            self.visit_statements(node.body)
            scope = self.scope
            if isinstance(context, ContextManager):
                defined: Dict[str, IRHandle] = {}
                redefined: Dict[str, IRHandle] = {}
                restore_types: Dict[str, Type[IRValue]] = {}
                for name in scope.defined:
                    value = materialize_ir_value(scope.lookup(name))
                    defined[name] = value.to_ir()
                    restore_types[name] = type(value)
                for name in scope.redefined:
                    value = materialize_ir_value(scope.lookup(name))
                    redefined[name] = value.to_ir()
                    restore_types[name] = type(value)
                context.handle_yieldables(defined, redefined)
            context.__exit__(None, None, None)
        if isinstance(context, ContextManager):
            for name, handle in context.map_to_restore().items():
                self.scope.save(name, restore_types[name].from_ir(handle))
