.. Copyright (c) 2026 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

Runtime API
===========


JIT decorator
-------------

.. autodecorator:: asc.experimental.asctile.jit


JIT options
-----------

Attributes from the following classes can be used as keyword arguments in :py:obj:`asc.experimental.asctile.jit` decorator.

.. tip::

    The user can mix options from different sections. For example, to disable ``CodegenOptions.capture_exceptions`` and
    set ``CompileOptions.opt_level`` to ``2``, the kernel function may be decorated as the following:

    .. code-block:: python

        @asctile.jit(capture_exceptions=False, opt_level=2)
        def kernel(x, y):
            ...

.. autoclass:: asc.CodegenOptions
   :members:
   :member-order: bysource

.. autoclass:: asc.experimental.asctile.CompileOptions
   :members:
   :inherited-members:
   :member-order: bysource

.. autoclass:: asc.LaunchOptions
   :members:
   :member-order: bysource


Auto-tuning
-----------

.. autodecorator:: asc.experimental.asctile.autotune


Platform configuration
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   asc.experimental.asctile.Backend
   asc.experimental.asctile.Platform


.. autofunction:: asc.experimental.asctile.set_platform
