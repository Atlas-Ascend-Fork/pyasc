# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Warmup samples must never become the reported median."""
import pytest

from asc.lib.profiling.result import ProfilingTask, task_time_median


def tasks():
    return [
        ProfilingTask(0, "kernel", "AI_CORE", 99.),
        ProfilingTask(1, "kernel", "AI_CORE", 3.),
        ProfilingTask(2, "kernel", "AI_CORE", 1.)
    ]


@pytest.mark.parametrize("skip", [3, 4])
def test_all_samples_are_warmup(skip):
    with pytest.raises(RuntimeError, match="no timings"):
        task_time_median(tasks(), skip=skip)


def test_single_warmup_sample_is_not_reported():
    with pytest.raises(RuntimeError, match="no timings"):
        task_time_median(tasks()[:1], skip=1)


def test_negative_warmup_count_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        task_time_median(tasks(), skip=-1)


@pytest.mark.parametrize("skip,expected", [(0, 3.), (1, 2.), (2, 1.)])
def test_remaining_sample_median(skip, expected):
    assert task_time_median(iter(tasks()), skip=skip) == expected


def test_filtering_happens_before_warmup_exclusion():
    records = [ProfilingTask(0, "copy", "AICPU", 1000.), ProfilingTask(1, "other", "AI_CORE", 1000.), *tasks()]
    assert task_time_median(records, name="kernel", skip=1) == 2.
