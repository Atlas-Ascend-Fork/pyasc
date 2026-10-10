.. Copyright (c) 2026 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

Tools
=====

The following tools are provided to project contributors to enhance their development and debugging capabilities.


MLIR LSP server: ``asctile-lsp``
--------------------------------

The tool implements language server protocol which is used by IDEs to effectively provide syntax highlighting, as well
as other language processing features, both for built-in MLIR dialects and AscIR extensions.

To enable ``asctile-lsp`` server in Visual Studio Code:

1. Install `MLIR extension <https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-mlir>`__.
2. Obtain full path to the built executable (e.g. run ``which asctile-lsp``).
3. Paste the path to *Mlir: Server_path* setting (``mlir.server_path``).


MLIR optimizer driver: ``asctile-opt``
--------------------------------------

The tool supports all features and command line options that are supported by ``mlir-opt`` (LLVM built-in application),
and is also able to run AscIR passes with its dialects and extensions.

For example:

.. code-block:: bash

    asctile-opt -ascendc-insert-bufid-sync -canonicalize -mlir-print-ir-before-all test.mlir
