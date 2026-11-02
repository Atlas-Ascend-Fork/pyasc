.. Copyright (c) 2026 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

Setup runtime environment
=========================

After the :doc:`installation <install-from-pypi>`, the following dependencies should be installed before running PyAsc operators:

.. contents::
    :depth: 1
    :local:

Ascend CANN
-----------

1. Download installation package from `the download center <https://www.hiascend.com/cann/download>`__
   (recommended version is ``9.2.0``) **depending on the host platform**:

   .. code:: bash

      wget https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/CANN/CANN%209.2.0/Ascend-cann_9.2.0_linux-x86_64.run

2. Install downloaded package with the following command:

   .. code:: bash

      bash Ascend-cann_9.2.0_linux-x86_64.run --full

3. Enable CANN environment:

   .. code-block:: bash

       source /usr/local/Ascend/cann/set_env.sh  # or ~/Ascend/cann/set_env.sh in case of user installation
       export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH

To learn more about CANN environment, visit `the homepage <https://www.hiascend.com/cann>`__.
Ascend C API reference is available in `the documentation <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920/API/ascendcopapi/atlasascendc_api_07_0003.html>`__.

PyTorch
-------

It is recommended to install ``torch`` (CPU-only) to run tutorials and develop PyAsc operators.
The package can be installed via pip:

.. code-block:: bash

   python3 -m pip install "torch>=2.7.1" --index-url https://download.pytorch.org/whl/cpu

Visit `the official website <https://pytorch.org/get-started/locally/>`__ to learn more.

Additionally, ``torch_npu`` may be installed to allocate host-side tensors directly on Ascend hardware (if available).
The setup commands are available on `Ascend for PyTorch <https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download>`__ webpage.
