.. Copyright (c) 2026 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

Profiling
=========

.. contents:: Table of Contents
   :local:

``msprof``-backed profiler
--------------------------

Use ``asc.lib.profiling`` to collect CANN-native profiling data for kernel launches without external tooling.
The profiler is backed by ``libmsprofiler.so`` and exports task timing data as ``ProfilingResult`` objects.

Profiling requires kernel execution on an NPU. The ``Model`` simulator backend is not supported.
See also: :py:func:`asc.experimental.asctile.set_platform`.

The main API is:

- ``Profiler(result_path=None)``: creates a profiler instance. When ``result_path`` is omitted, a temporary directory is
  used and removed after parsing.
- ``Profiler.profile(...)``: context manager that surrounds one or more kernel launches. It starts and stops profiling,
  exports CANN reports, parses the latest result, and stores it in ``profiler.last_result``.
- ``ProfilingResult.tasks``: tuple of ``ProfilingTask`` objects parsed from one profiling run.
- ``ProfilingTask``: task ID, kernel name, task type, and duration in microseconds.
- ``task_time_median(tasks, name=None, skip=0)``: a helper function that returns the median duration in microseconds for
  AI Core tasks.

For stable timing, launch the kernel several times inside a single ``profile()`` context and exclude the first launch
from the median calculation, as shown in the following example:

.. code-block:: python

    import asctile
    from asc.lib.profiling import Profiler, task_time_median

    @asctile.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, size: int):
        ...

    asctile.set_platform(asctile.Backend.NPU)

    profiler = Profiler(result_path=".")  # keeping PROF_* outputs in the current working directory
    with profiler.profile():
        for _ in range(10):  # adjust the run count for your task
            add_kernel[block_num](x, y, out, size)

    median_us = task_time_median(profiler.last_result.tasks, skip=1)
    print(f"add_kernel median task time: {median_us} μs")

All launches that should contribute to the reported timing must occur inside the context. It is strongly recommended to
keep input initialization and correctness checks outside the context. The first profiled launch acts as a warmup sample
and is excluded from the median by ``skip=1``.

API reference
-------------

.. currentmodule:: asc.lib.profiling

.. autoclass:: Profiler
   :members: last_result, profile

.. autosummary::
    :toctree: generated
    :nosignatures:

    AicoreMetrics
    ProfileType
    ProfilingResult
    ProfilingTask

.. autofunction:: task_time_median
