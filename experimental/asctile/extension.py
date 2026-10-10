# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Iterable, Mapping

local_package_dirs = {
    "asctile": "experimental/asctile/python/asctile",
    "asctile.language": "experimental/asctile/python/asctile/language",
    "asctile.runtime": "experimental/asctile/python/asctile/runtime",
}


def get_packages(prefix: str) -> Iterable[str]:
    return (f"{prefix}.{pkg}" for pkg in local_package_dirs.keys())


def get_package_dirs(prefix: str) -> Mapping[str, str]:
    return {f"{prefix}.{pkg}": path for pkg, path in local_package_dirs.items()}


def get_devtools(cmake_dir: str) -> Mapping[str, str]:
    return {
        "asctile-lsp": f"{cmake_dir}/bin/asctile-lsp",
        "asctile-opt": f"{cmake_dir}/bin/asctile-opt",
    }
