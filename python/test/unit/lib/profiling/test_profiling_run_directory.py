# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Select a run directory rather than unrelated top-level files."""
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


@pytest.mark.parametrize("extra_file", ["z-notes.txt", "zz.json"])
def test_top_level_file_does_not_hide_the_latest_run(profiler, extra_file):
    write_run(profiler, "PROF_20261001")
    write_run(profiler, "PROF_20261002")
    (module.Path(profiler.result_path) / extra_file).write_text("notes")
    profiler.store_last_result()
    assert profiler.last_result.run_id == "PROF_20261002"
    assert profiler.last_result.tasks[0].duration == 3.5


def test_only_regular_files_reports_no_runs(profiler):
    (module.Path(profiler.result_path) / "notes.txt").write_text("notes")
    with pytest.raises(RuntimeError, match="There is no profiling results"):
        profiler.store_last_result()


def test_latest_directory_is_still_selected_without_extra_files(profiler):
    write_run(profiler, "PROF_20261001")
    write_run(profiler, "PROF_20261002")
    profiler.store_last_result()
    assert profiler.last_result.run_id == "PROF_20261002"
