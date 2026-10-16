# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Native session boundaries are recorded; Profiler state and CSV parsing are real."""
import csv

import pytest

from asc.lib.profiling import profiler as module


class NativeRecorder:

    def __init__(self):
        self.events = []

    def init(self, path):
        self.events.append("init")

    def create_config(self, *args):
        self.events.append("create")
        return object()

    def start(self, config):
        self.events.append("start")

    def stop(self, config):
        self.events.append("stop")

    def destroy_config(self, config):
        self.events.append("destroy")

    def finalize(self):
        self.events.append("finalize")


@pytest.fixture
def profiler(tmp_path, monkeypatch):
    native = NativeRecorder()
    monkeypatch.setattr(module, "MsprofInterface", lambda: native)
    instance = module.Profiler(str(tmp_path))
    run = tmp_path / "run_1"
    run.mkdir()
    with (run / "op_summary_1.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["Task ID", "Op Name", "Task Type", "Task Duration(us)"])
        writer.writerow([1, "kernel", "AI_CORE", 3.5])
    monkeypatch.setattr(instance, "export", lambda: native.events.append("export"))
    original_cleanup = instance.cleanup

    def cleanup():
        native.events.append("cleanup")
        assert not instance.started
        original_cleanup()

    monkeypatch.setattr(instance, "cleanup", cleanup)
    return instance, native


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_body_exception_finalizes_before_cleanup(profiler, error):
    instance, native = profiler
    with pytest.raises(error, match="body failure"):
        with instance.profile(device_id=0):
            raise error("body failure")
    assert native.events == ["init", "create", "start", "stop", "destroy", "finalize", "cleanup"]
    assert instance.config is None
    assert not instance.started


def test_repeated_stop_does_not_reuse_destroyed_config(profiler):
    instance, native = profiler
    instance.start(device_id=0)
    instance.stop()
    instance.stop()
    assert native.events == ["init", "create", "start", "stop", "destroy", "finalize"]
    assert not instance.started
    assert instance.config is None


def test_successful_context_parses_csv_after_finalization(profiler):
    instance, native = profiler
    with instance.profile(device_id=0):
        pass
    assert native.events == ["init", "create", "start", "stop", "destroy", "finalize", "export", "cleanup"]
    assert instance.last_result.tasks[0].duration == 3.5
    assert not instance.started
