.. Copyright (c) 2026 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

AI agent skills for Ascend C
============================

`CANNBot repository <https://gitcode.com/cann/cannbot-skills>`__ provides skills
for AI agents with specialized knowledge about Ascend C API, NPU architecture,
Tiling design, and Matmul/GEMM optimization. These skills give the AI coding assistant
context-aware help with operator development, API usage, and explaining Ascend C specifics.
The following instructions are recommended to use with OpenCode client.


Recommended Skills
------------------

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Skill
      - Purpose
    * - ``ascendc-api-best-practices``
      - Comprehensive reference for all Ascend C API categories: arithmetic, reduction, sort,
        DataCopy, Transpose, Buffer management, precision conversion, pipeline operations,
        and API restrictions
    * - ``ascendc-blaze-best-practice``
      - Matmul/Cube/GEMM/BMM development via Blaze/tensor_api path. Covers AIC/StreamK/FixpOpti
        modes, dispatch modes, Tiling strategies, and workspace configuration
    * - ``ascendc-code-review``
      - Code review pipeline for Ascend C: file review, PR review, large PR handling,
        design consistency checks. Includes 11 reference rule sets covering API usage,
        performance, security, SIMT constraints, and C++ coding standards
    * - ``ascendc-crash-debug``
      - Debugging methodology for hang/crash/deadlock scenarios: symptom-to-cause mapping,
        Buffer deadlock diagnosis, cross-core sync issues, memory error detection via
        mssanitizer. Includes crash workflows and memcheck references
    * - ``ascendc-docs-search``
      - Local search across API documents and code examples from ``asc-devkit``.
        Falls back to online Huawei Ascend Community search when local resources are insufficient
    * - ``ascendc-performance-best-practices``
      - Performance optimization patterns by operator family: MatMul, RadixSort, Scalar,
        Reduction, Elementwise. Covers pingpong, swap, streamk, fullload, and 8 optimization rules
    * - ``ascendc-perf-optimize``
      - 4-step performance optimization strategy: Tiling modeling → inter-card pipeline →
        inter-core pipeline → intra-core pipeline. Bound diagnosis (compute/memory/communication)
        with Tiling parameter refinement. Self-contained tiling models for all operator categories
    * - ``ascendc-precision-debug``
      - Systematic precision issue diagnosis: pipeline sync (EnQue/DeQue), DataCopy alignment,
        Cast RoundMode, FP16/BF16 cross-validation, DumpTensor 7-step method. Includes error
        analysis scripts and boundary test generation
    * - ``ascendc-runtime-debug``
      - Runtime error debugging: comprehensive error code reference (161xxx/361xxx/561xxx),
        kernel binary debugging (compilation, cache, SEL matching, opParaSize), plog analysis.
        Includes debug workflows and error code tables
    * - ``ascendc-tiling-design``
      - Tiling methodology for all operator categories: Reduction, Sort, Elementwise, Broadcast,
        MatMul, Conv. Covers UB budgeting, Buffer strategies, and chunk sizing
    * - ``npu-arch``
      - NPU architecture knowledge: DAV_2201 vs DAV_3510, buffer sizes (UB/L0C/BT),
        Cube/Vector/MTE units, SIMD vs SIMD-RegBase, NDDMA, CCU
    * - ``ascendc-regbase-best-practice``
      - DAV_3510 RegBase (SIMD-RegBase) development: API constraints, implementation
        structure, common pitfalls, and real reference operators for the
        ``RegTensor`` / ``MaskReg`` / ``asc_vf_call`` / ``__simd_vf__`` paradigm
    * - ``ascendc-sync-audit``
      - Ascend C signal synchronization verification and fix: detects missing/mismatched
        sync between pipelines (MTE2/MTE3/Vector/Cube) and across cores
        (CrossCoreSetFlag/SyncAll), flag reuse conflicts, and buffer-index
        inconsistencies that cause hangs, deadlocks, or half-finished data; ships
        static analyzers (``sync_audit.py``, ``ascendc_flow_analyzer.py``) plus a
        333-PR case retriever for fix patterns
    * - ``ops-profiling``
      - On-board performance collection and analysis via ``msprof``: standard /
        compare / quick / batch modes (``msprof_profile_run.sh``), bottleneck
        diagnosis (CUBE/VECTOR/MTE2/MTE1/FIXPIPE/SCALAR); MC2/multi-rank (fork)
        operators must use ``msprof`` (not ``msprof op``)
    * - ``ops-simulator``
      - CANN Simulator (``npusim``, formerly ``cannsim``) for functional and
        performance simulation without NPU hardware: Ascend 950 only, single-card,
        AI Core compute only (no MC2/HCCL); ``summary.json`` quick diagnosis and
        trace bubble analysis with PC-to-source mapping
    * - ``aiss-tiling-solver``
      - ``TilingSolver`` CLI for automatic optimal tiling of MatMul / Vector
        operators (Z3-based): ``platform_info`` collects hardware params
        (L0A/L0B/L0C/L1/UB/L2/HBM sizes, bandwidths, CUBE_OPS_PER_CYCLE) and the
        solver returns base/single-core tile sizes and double-buffer flags
    * - ``ops-precision-standard``
      - Operator precision standards (mixed tolerance atol/rtol) per dtype and
        operator category: random / non-compute / integer / quantization / float;
        used for ST verification and FP16/BF16/FP32 acceptance criteria


Directory Structure
-------------------

The ``cannbot-skills`` repository should be cloned as a **sibling directory** to ``pyasc`` (not inside the project).
Skills are symlinked into ``.opencode/skills/``, the ``ops-code-reviewer`` agents into ``.opencode/agents/``,
and ``asc-devkit`` is symlinked in the project root.
All paths use the ``$CANNBOT_SKILLS`` environment variable.


Installation
------------

.. code-block:: bash

    # Clone cannbot-skills as sibling to pyasc
    export CANNBOT_SKILLS=$PWD/../cannbot-skills
    git clone https://gitcode.com/cann/cannbot-skills.git $CANNBOT_SKILLS

    # Clone asc-devkit (API docs + examples)
    git clone https://gitcode.com/cann/asc-devkit.git $CANNBOT_SKILLS/plugins-official/ops-direct-invoke/asc-devkit

    # Create symlinks
    skills=(
        ascendc-api-best-practices
        ascendc-blaze-best-practice
        ascendc-code-review
        ascendc-crash-debug
        ascendc-docs-search
        ascendc-performance-best-practices
        ascendc-perf-optimize
        ascendc-precision-debug
        ascendc-runtime-debug
        ascendc-tiling-design
        npu-arch
        ascendc-regbase-best-practice
        ascendc-sync-audit
        ops-profiling
        ops-simulator
        aiss-tiling-solver
        ops-precision-standard
    )
    for skill in "${skills[@]}"; do
        ln -sfn $CANNBOT_SKILLS/ops/$skill .opencode/skills/$skill
    done
    ln -sfn $CANNBOT_SKILLS/plugins-official/ops-direct-invoke/asc-devkit ./asc-devkit

    # ops-code-reviewer plugin: agent symlinks only (do not overwrite AGENTS.md)
    mkdir -p .opencode/agents
    for agent in ascendc-code-summarizer ascendc-ops-reviewer; do
        ln -sfn $CANNBOT_SKILLS/plugins-official/ops-code-reviewer/agents/$agent.md .opencode/agents/$agent.md
    done

    # Verify skills installation
    for s in .opencode/skills/*/; do
      [ -f "$s/SKILL.md" ] && echo "OK: $s" || echo "MISSING: $s"
    done

Restart OpenCode to load the skills.

Hiding Symlinks from Git
~~~~~~~~~~~~~~~~~~~~~~~~

The symlinks created above (``asc-devkit``, skills under ``.opencode/skills/``, and agents under
``.opencode/agents/``) are external dependencies and should not be tracked by git. It is recommended
to add them to the local ``.git/info/exclude`` file:

.. code-block:: bash

    echo "
    # CANNBot skills symlinks
    asc-devkit
    .opencode/skills/ascendc-*
    .opencode/skills/npu-arch
    .opencode/skills/ops-*
    .opencode/skills/aiss-*
    .opencode/agents/ascendc-*" >> .git/info/exclude


Updating
--------

.. code-block:: bash

    cd $CANNBOT_SKILLS && git pull
    cd plugins-official/ops-direct-invoke/asc-devkit && git pull

Restart OpenCode to apply updates.


Alternative: Automated Installation
-----------------------------------

Use ``init.sh`` from a plugin directory to install all skills from that plugin:

.. code-block:: bash

    cd $CANNBOT_SKILLS/plugins-official/<plugin-name>
    bash init.sh project opencode /path/to/pyasc

See the `cannbot-skills README <https://gitcode.com/cann/cannbot-skills>`__ for available plugins.
