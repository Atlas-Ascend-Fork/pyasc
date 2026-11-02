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
