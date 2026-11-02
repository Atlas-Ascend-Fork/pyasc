.. Copyright (c) 2026 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

Coding style conventions
========================

This document outlines the coding style conventions for the project. These guidelines are primarily based on the LLVM coding style with a few exceptions and project-specific tweaks. Below are the main rules and conventions:

General conventions for C++ files
---------------------------------

1. **File Extensions**:

   - Header files must have the ``.h`` extension.
   - Implementation (library) files must have the ``.cpp`` extension.

2. **Include Guards**:

   - All header files must include traditional include guards (e.g., ``#ifndef HEADER_H``, ``#define HEADER_H``, ``#endif``).
   - **Note**: The use of ``#pragma once`` is **not allowed**.

3. **Indentation**: The code should use **4 spaces** for indentation (no tabs). ``case``/``default`` labels should be indented relative to the ``switch`` statement. Access specifiers (``public``/``protected``/``private``) are aligned with the enclosing ``class``/``struct`` declaration (i.e., not indented relative to it).

4. **Naming Conventions**:

   - **PascalCase**: Used for naming types such as classes, structs, enums, and typedefs. Example: ``MyClass``, ``MyEnum``.
   - **camelCase**: Used for variables (local and global), functions, and class members. Example: ``myFunction``, ``myVariable``.
   - **UPPER_SNAKE_CASE**: Typically used for macros. Example: ``MY_CONSTANT``, ``MAX_BUFFER_SIZE``.
   - **kebab-case**: Not commonly used in code, but may be used in filenames for tests and other resources. Example: ``my-test-case``.

5. **Namespace Declarations**:

   - After closing a namespace, add a comment ``// namespace <namespace_name>``. If the namespace is anonymous, omit the name.
   - **Note**: Do not use ``using namespace`` in header files.

6. **Brace Placement**: Opening brace on a new line for function definitions; on the same line for classes/structs/enums/namespaces and for control statements (``if``/``for``/``while``/``switch``). ``else`` and ``catch`` remain on the same line as the preceding closing brace. Empty function definitions may be kept on a single line.

7. **Short Statements**: Single-line ``if`` statements, loops, and ``case`` labels are not allowed. Empty blocks and empty/short inline function definitions may be kept on a single line.

8. **Spacing and Alignment**: Pointers/references aligned left (e.g., ``int* ptr``). Space before control statement parentheses and before C++11 braced lists. No space in empty parentheses, C-style casts, regular parentheses, or square brackets. Exactly one space before trailing line comments. Operands of multi-line expressions and trailing comments are aligned. When the arguments of a call or declaration do not fit on one line, the line is broken right after the opening bracket.

9. **Templates**: Always break before template declarations.

10. **Line Length**: Maximum 120 characters.

11. **Macro Definitions**:

   - Defining macros in header files is **discouraged**, unless absolutely necessary.
   - To define a global constant, ``constexpr`` syntax and **camelCase** naming should be used.

12. **Include Ordering**:

   - ``#include`` statements are regrouped and sorted automatically by ``clang-format`` (see ``experimental/asctile/.clang-format``). The groups, in order, are:

     1. ``asctile/`` project includes.
     2. ``ascir/`` project includes.
     3. MLIR, Clang, and LLVM includes (MLIR headers are sorted before Clang/LLVM headers within the group).
     4. ``pybind11/`` includes.
     5. Standard library includes (``<optional>``, ``<vector>``, etc.).
     6. Other local includes (quoted includes not matching any of the above, e.g., project-internal headers like ``"Common.h"``).

   - Each group is separated by an empty line. Within each group, includes are sorted alphabetically by file name.
   - Example:

     .. code-block:: cpp

        #include "asctile/Conversion/LowerToAsc/Passes.h"

        #include "ascir/Dialect/Asc/IR/Asc.h"
        #include "ascir/Dialect/Utils/ConstantOpBuilder.h"

        #include "mlir/Dialect/Arith/IR/Arith.h"
        #include "mlir/Dialect/Func/IR/FuncOps.h"

        #include "Common.h"


13. **Anonymous Namespace**: If a class or function is defined and declared in a ``.cpp`` file but not used elsewhere in the project, it should be placed inside an anonymous namespace. It should **not** be marked as ``static``.

14. **Template Argument Naming**:

   - Template typename arguments should follow **PascalCase**. Example: ``typename AttrT``.
   - Non-type template arguments should follow **camelCase**. Example: ``size_t size``.
   - Always use ``typename`` instead of ``class`` for template arguments.

General conventions for Python files
------------------------------------

The Python codebase follows PEP 8 with the project-specific tweaks. The main rules are:

1. **File Header**: Every Python file must start with the project copyright/license header (the same header used for C++ files).

2. **Line Length**: Maximum **120 characters** (applied consistently by ``yapf`` and ``ruff``).

3. **Indentation**: **4 spaces**, no tabs.

4. **Naming Conventions**:

   - **PascalCase**: Classes, dataclasses, exceptions, type aliases, and enums. Example: ``LocalTensor``, ``CompileOptions``, ``RoundMode``.
   - **snake_case**: Functions, methods, variables, and module-level constants. Example: ``ceildiv``, ``constant_tile``, ``all_dtypes``.
   - Single uppercase letter for ``TypeVar`` parameters (e.g., ``T``).

5. **Imports**: Imports are organized into three groups, separated by a blank line:

   1. Standard library imports (e.g., ``from typing import ...``).
   2. Project and third-party imports (``asc.*``, ``pybind11``, and other external packages).
   3. Local relative imports (e.g., ``from .local_tensor import LocalTensor``).

   - Imports within each group are sorted alphabetically. Wildcard imports (``import *``) are not allowed.

6. **Type Hints**: Public functions and methods must be annotated (PEP 484). Use ``from __future__ import annotations`` where forward references are needed, and declare overloads with ``@overload``.

7. **Docstrings**: Public functions, classes, and modules should have triple-quoted docstrings. Use Google-style sections (``Args``, ``Returns``, ``Raises``, ``Examples``) for non-trivial callables, and include runnable usage examples for user-facing APIs.

8. **String Quotes**: Double quotes (``"..."``) are preferred for string literals and f-strings.

9. **Testing**: Tests use ``pytest`` and live under the ``test/`` directory. Test files are named ``test_*.py`` and test functions are named ``test_*``. Shared fixtures and helpers go in ``conftest.py`` or ``helpers.py``.

10. **Error Handling**: Raise specific exceptions (e.g., ``ValueError``, ``RuntimeError``, ``TypeError``) with descriptive messages. Prefer exceptions over returning error codes.

Conventions for MLIR dialects
-----------------------------

Definitions of operations, types, attributes, interfaces, and other entities should be sorted alphabetically within the corresponding TableGen file.

Conventions for MLIR passes
---------------------------

1. **Pass File Organization**: Each MLIR pass should be placed in a **separate** ``.cpp`` file under the ``Transforms`` directory, within the directory corresponding to the specific MLIR dialect.

2. **File Naming**: The name of the ``.cpp`` file should match the name of the pass **without the "Pass" suffix**. For example, the file for the ``FoldVariablePass`` pass should be named ``FoldVariable.cpp``.

3. **Pass Declarations**: In the ``Passes.td`` file, in ``CMakeLists.txt``, and in the constructor functions header ``Passes.h``, pass names should be listed in **alphabetical order**.

Conventions for LIT tests
-------------------------

1. **Test Directory Structure**:

   - Tests are placed in the ``test`` directory.
   - Filenames for test files should use **kebab-case**. Example: ``my-test-case.mlir``.

2. **Test File Format**: The first line of the test file should generally contain one or more ``// RUN:`` commands that specify how the test should be executed.

3. **Test File Organization**:

   - When adding a new MLIR operation, type, or attribute, a test for that feature should be added under the ``IR`` directory within the appropriate dialect's directory.
   - When adding a new MLIR pass, a set of tests (typically multiple ``func.func`` operations) should be added under the ``Transforms`` directory of the corresponding dialect where the pass is introduced. Name of file should correspond to a pass name.
   - For the emission of a new operation, a test should be placed under the ``Target`` directory.
   - End-to-end tests for tools should be located under the ``Tools`` directory.

Additional considerations
-------------------------

- **Readability**: Code should be written for **clarity and maintainability**, not just brevity. Use meaningful names for functions, variables, and types. Comment complex or non-obvious code to aid future developers.

- **Refactoring**: Refrain from making large, non-essential refactoring changes in areas that are not directly related to the task at hand. Always aim for minimal disruption in the codebase.

- **Version control**:

  - Since commits are squashed into a single commit during the merge, detailed commit messages are not required. However, if needed, include helpful information about the change.
  - Pull request titles should be clear and describe the intention behind the changes. Follow conventional commit styles where possible.
  - Pull request description is optional if the title is sufficiently explanatory. It should be added in case of complex changes to clearly describe what has been implemented.
  - Make sure the code builds and passes tests locally before committing.

Code formatting and static analysis tools
-----------------------------------------

To help ensure that the code adheres to the coding style conventions automatically, it is strongly recommended using the following tools:
``clang-format``, ``clang-tidy``, ``yapf``, ``ruff``.

Run linting before every commit using the ``pre-commit`` hooks:

.. code-block:: cpp

   python3 -m pip install pre-commit
   pre-commit install

Install and configure Visual Studio Code extensions to format files on save:

- `Clang-Format extension <https://marketplace.visualstudio.com/items?itemName=xaver.clang-format>`__
- `yapf extension <https://marketplace.visualstudio.com/items?itemName=eeyore.yapf>`__

This style guide serves to maintain consistency across the codebase, making it easier to read, maintain, and extend. Adhering to these conventions will improve collaboration, reduce errors, and make it easier for new contributors to get up to speed.
