# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import types
from typing import Any, Dict, Optional, Union

overrides: Dict[str, Any] = {}


def override(obj: Union[Any, str], name: Optional[str] = None) -> Any:
    if isinstance(obj, str):
        return lambda target: override(target, obj)
    key = name or obj.__name__
    if overrides.setdefault(key, obj) is not obj:
        raise RuntimeError(f"'{key}' is already overridden")
    return obj


def make_module() -> types.ModuleType:
    import asc.experimental.asctile as asctile
    module = types.ModuleType("asctile")
    for name in asctile.__all__:
        value = getattr(asctile, name)
        if hasattr(value, "__wrapped__"):
            continue
        setattr(module, name, value)
    for name, value in overrides.items():
        setattr(module, name, value)
    return module


overrides_by_function: Dict[Any, Any] = {}


def to_interpreted(value: Any) -> Any:
    if not isinstance(value, types.FunctionType):
        return value
    if not overrides_by_function:
        import asc.experimental.asctile as asctile
        for name in asctile.__all__:
            if name in overrides:
                overrides_by_function[getattr(asctile, name)] = overrides[name]
    return overrides_by_function.get(value, value)


def make_asc_module() -> types.ModuleType:
    import asc
    module = types.ModuleType("asc")

    def __getattr__(name: str) -> Any:
        return getattr(asc, name)

    def sync_all(*args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("'sync_all' is not supported by the interpreter")

    module.__getattr__ = __getattr__
    module.sync_all = sync_all
    return module
