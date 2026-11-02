.. Copyright (c) 2026 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

Debugging
=========

.. contents:: Table of Contents
   :local:

.. currentmodule:: asc.experimental.asctile

AscTile provides four debug operations to inspect values and check assumptions while developing kernels.
They fall into two groups, depending on **when** they run:

=========== =================================== ============================================== =======================
Group       Operations                          When they run                                  Needs a running kernel?
=========== =================================== ============================================== =======================
Device-side ``device_print``, ``device_assert`` During kernel execution on the NPU             Yes
Host-side   ``static_print``, ``static_assert`` During JIT compilation, before the kernel runs No
=========== =================================== ============================================== =======================

Use the **device-side** operations for runtime values that only exist on the device (a loaded tile, a
computed scalar), and the **host-side** operations for compile-time values (a ``ConstExpr`` argument, a
tile shape, a dtype) — they are cheaper and work without hardware.


Enabling device-side debugging
------------------------------

Device-side operations are **off by default**. To turn them on, decorate the kernel with ``debug=True``:

.. code-block:: python

   @asctile.jit(debug=True)
   def kernel(x_ptr: asctile.GlobalAddress, size: int):
       ...

When ``debug`` is not set (the default), ``device_print`` and ``device_assert`` do nothing and add no
overhead to your kernel — keep it this way for production runs.

.. warning::

   ``debug=True`` may cause performance regressions. Enable it only while developing or validating correctness, and
   switch it off for production builds.

Device-side output is produced only when the kernel is actually executed — on the NPU or the Model
simulator; nothing is printed if the kernel is compiled but never launched. The host-side operations
(``static_print``, ``static_assert``) are different: they run during JIT compilation, before the kernel
runs, so they take effect even before the kernel is launched and without any hardware.


Device-side operations
----------------------

These run during kernel execution on the device.

.. autofunction:: device_print
   :no-index:

.. autofunction:: device_assert
   :no-index:


Host-side operations
--------------------

These run during JIT compilation, before the kernel runs, and work without any hardware.

.. autofunction:: static_print
   :no-index:

.. autofunction:: static_assert
   :no-index:


Environment variables
---------------------

These environment variables complement the debug operations and control what the toolchain keeps on
disk and whether a compiled kernel is actually launched.

``PYASC_DUMP_PATH=<dir>`` (optional)
    directory where the compiler keeps generated artifacts: intermediate IR (``.mlir``), Ascend C source
    (``.cpp``) and the object file (``.o``)

``CAMODEL_LOG_PATH=<dir>`` (optional)
    directory for the ``Model`` simulator logs; only used on the ``Model`` backend — setting it persists
    the logs instead of a temp directory removed on exit

``PYASC_DRY_RUN=1`` (optional)
    when set, the kernel is compiled but not launched on the device — useful to validate compilation
    without hardware

.. spoiler:: Additional debug capabilities

   **1. Capture build artifacts.**

   The ``PYASC_DUMP_PATH`` environment variable can be set to make the compiler keep generated files in a directory:

   .. code-block:: bash

      PYASC_DUMP_PATH=dumps python3 experimental/asctile/python/tutorials/01-vector-add.py

   The following directory structure is created in the current directory:

   .. code-block:: text

      dumps/
        ascendc.cpp  # generated Ascend C file
        ascir.mlir   # final IR which is used to emit Ascend C
        binary.o     # object file produced by the Bisheng compiler
        codegen.mlir # input IR captured from the JIT function

   .. note::

      Set the ``always_compile=True`` JIT option to avoid caching and trigger compilation each time.

   **2. Tune generated Ascend C code.**

   Ascend C code generated from AscTile can be injected back into AscTile Python code:

   - Pass the content of the generated kernel as the parameter to ``asctile.inline``.
   - Comment out the ``TPipe`` definition in the code, as shown below.

   .. code-block:: python

      @asctile.jit
      def vadd_kernel(x_ptr: asctile.GlobalAddress, y_ptr: asctile.GlobalAddress,
                      out_ptr: asctile.GlobalAddress, size: int,
                      tile_size: asctile.ConstExpr[int], tile_per_block: asctile.ConstExpr[int]):
          asctile.inline('''
          constexpr int32_t c32_i32 = 32;
          constexpr int64_t c128_i64 = 128;
          .       .       .
          constexpr int32_t c128_i32 = 128;
          // AscendC::TPipe v5;
          AscendC::GlobalTensor<float> v6;
          v6.SetGlobalBuffer(v1_x_ptr);
          .       .       .
            get_buf(PIPE_MTE3, 5, 0);
            AscendC::DataCopyPad(v55, v11, v58);
            rls_buf(PIPE_MTE3, 5, 0);
          }
          return;
              ''')

   **3. Pass arguments to inline code.**

   Kernel arguments can be passed to ``asctile.inline`` through the ``args`` parameter. Use ``$<index>`` placeholders,
   such as ``$0``, ``$1``, and ``$2``, to reference arguments by their position in the list:

   .. code-block:: python

      @asctile.jit
      def kernel(x_ptr: asctile.GlobalAddress, y_ptr: asctile.GlobalAddress,
                 out_ptr: asctile.GlobalAddress, size: int):
          asctile.inline('''
              auto input_x = $0;
              auto input_y = $1;
              auto output = $2;
              int64_t length = $3;
              AscendC::GlobalTensor<float> x_gm;
              x_gm.SetGlobalBuffer(input_x);
              AscendC::GlobalTensor<float> y_gm;
              y_gm.SetGlobalBuffer(input_y);
              AscendC::GlobalTensor<float> out_gm;
              out_gm.SetGlobalBuffer(output);
          ''', [x_ptr, y_ptr, out_ptr, size])

   Each argument is materialized as an IR value and substituted into the generated code at the corresponding placeholder
   position.
