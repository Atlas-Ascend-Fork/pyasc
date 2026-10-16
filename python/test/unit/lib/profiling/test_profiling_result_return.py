# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""The annotated result accessors must return the parsed result."""
import csv

import pytest

from asc.lib.profiling import profiler as module


@pytest.fixture
def profiler(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MsprofInterface", lambda: None)
    return module.Profiler(str(tmp_path))


def write_run(profiler, run_id, filename="op_summary_1.csv"):
    path = module.Path(profiler.result_path) / run_id
    path.mkdir()
    with (path / filename).open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["Task ID", "Op Name", "Task Type", "Task Duration(us)"])
        writer.writerow([1, "kernel", "AI_CORE", 3.5])


@pytest.mark.parametrize("accessor", ["store_result", "store_last_result"])
def test_result_accessor_returns_the_cached_object(profiler, accessor):
    write_run(profiler, "PROF_20261002")
    method = getattr(profiler, accessor)
    result = method("PROF_20261002") if accessor == "store_result" else method()
    assert isinstance(result, module.ProfilingResult)
    assert result is profiler.results["PROF_20261002"]
    assert result is profiler.last_result
    assert result.tasks[0].duration == 3.5
    assert result.run_id == "PROF_20261002"
