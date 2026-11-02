.. Copyright (c) 2026 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

Quick start
===========

.. contents:: Table of Contents
   :local:

Why use AscTile
---------------

.. include:: ../../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- motivation-start -->
   :end-before: <!-- motivation-end -->

Install required packages
-------------------------

1. :doc:`Install PyAsc + AscTile package <../installation/install-from-pypi>`.
2. :doc:`Install CANN packages <../installation/setup-runtime-env>` if they are not installed.

Learn AscTile with tutorials
----------------------------

The :doc:`Tutorials section <tutorials/index>` explains how to write an AscTile kernel,
organize host code, set up the runtime platform, and launch kernels on the NPU. Start with the vector
addition tutorial, then continue in order through the fused softmax and matrix multiplication examples.

Further reading
---------------

- :doc:`Debugging <../programming-guide/debugging>`
- :doc:`Profiling <../programming-guide/profiling>`
- :doc:`Kernel API <../python-api/language>`
