# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional, Tuple

import numpy as np

task_types = {"AI_CORE", "AIV_SQE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV", "KERNEL_AIVEC", "KERNEL_AICORE"}


@dataclass(frozen=True)
class ProfilingTask:
    """A single kernel / AI Core task parsed from msprof CSV output."""

    id: int
    name: str
    type: str
    duration: float


@dataclass(frozen=True)
class ProfilingResult:
    """Parsed tasks from one profiler export run."""

    tasks: Tuple[ProfilingTask]
    run_id: str
    stored_at: datetime


def task_time_median(tasks: Iterable[ProfilingTask], name: Optional[str] = None, skip: int = 0) -> float:
    """Return the median duration (us) of AI Core tasks.

    Args:
        tasks: Tasks from :class:`ProfilingResult`.
        name: If set, only tasks with this kernel name are included. If omitted,
            the first AI Core task name is used as the filter.
        skip: Number of leading matching samples to drop (warmup).
    """
    if skip < 0:
        raise ValueError("skip must be non-negative")
    values = []
    for task in tasks:
        if task.type not in task_types:
            continue
        if name is None:
            name = task.name
        elif task.name != name:
            continue
        values.append(task.duration)
    values = values[skip:]
    if len(values) == 0:
        raise RuntimeError(f"There is no timings for task {name!r}")
    return np.round(np.median(np.array(values, dtype=np.float64)), 3).item()
