.. Copyright (c) 2026 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

Install from PyPI
=================

**Python 3.9 or newer** with pip is required for PyAsc.

.. spoiler:: Using a virtual environment (optional)

    It is recommended to activate `a virtual environment <https://docs.python.org/3/tutorial/venv.html>`__ (venv) instead of using a system-wide installation:

    .. code-block:: bash

        python3 -m venv $HOME/.pyasc_venv --prompt pyasc
        source $HOME/.pyasc_venv/bin/activate


.. attention::

    Latest **PyAsc + AscTile** package cannot co-exist with an outdated ``pyasc`` installation, so it should be uninstalled preemptively:

    .. code-block:: bash

        python3 -m pip uninstall --yes pyasc


Run the following command to download and install (or update) a pre-built Python package into the environment:

.. code-block:: bash

    python3 -m pip install --pre --upgrade pyasc-dev


After the installation, it is recommended to :doc:`setup runtime environment <setup-runtime-env>` in order to run PyAsc operators.
