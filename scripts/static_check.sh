#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -e

# ============================================================================
# Global Variables
# ============================================================================
SCRIPT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
COMPILE_DB=$(find "./" -name "compile_commands.json" -type f 2>/dev/null | head -n 1)
BASE_BRANCH="origin/master"

# clang-tidy diagnostic ignore list.
# Format: <path-suffix>|<symbol>|<check-name>
# A diagnostic block is dropped when its source path ends with <path-suffix>,
# the message references <symbol> (in single quotes), and the check name matches.
TIDY_IGNORE_LIST=(
    # extern "C" ABI entry, must keep PascalCase for the Python runtime to link.
    "python/asc/lib/runtime/print_utils.cpp|PrintWorkSpace|readability-identifier-naming"
)

# Counters
TIDY_ERROR_COUNT=0
TIDY_WARNING_COUNT=0

# ============================================================================
# Compile Flags Setup
# ============================================================================
MANUAL_COMPILE_FLAGS="-std=c++17 -I${PROJECT_ROOT}/include"

setup_compile_flags() {
    if [[ -n "$LLVM_INSTALL_PREFIX" && -d "$LLVM_INSTALL_PREFIX/include" ]]; then
        MANUAL_COMPILE_FLAGS="$MANUAL_COMPILE_FLAGS -I${LLVM_INSTALL_PREFIX}/include"
    fi

    local gcc_version=$(gcc -dumpversion 2>/dev/null || echo "9")
    local arch=$(uname -m)
    local triple="${arch}-linux-gnu"

    for dir in \
        "/usr/lib/gcc/${triple}/${gcc_version}/include" \
        "/usr/include/c++/${gcc_version}" \
        "/usr/include/c++/${gcc_version}/backward" \
        "/usr/include/${triple}/c++/${gcc_version}" \
        "/usr/include/${triple}/c++/${gcc_version}/bits"; do
        [[ -d "$dir" ]] && MANUAL_COMPILE_FLAGS="$MANUAL_COMPILE_FLAGS -I$dir"
    done
}

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" >&2
}

check_and_install_tools() {
    log "[INFO] Installing clang tools..."

    apt-get update -qq 2>&1 || true
    apt-get install -y clang clang-format clang-tidy clang-tools 2>&1

    local tools="clang-format clang-tidy clang-format-diff clang-tidy-diff"
    for tool in $tools; do
        command -v "$tool" &> /dev/null || { log "[ERROR] $tool installation failed"; exit 1; }
    done

    log "[INFO] clang tools installed successfully"
}

# ============================================================================
# Core Check Functions
# ============================================================================

run_clang_tidy_diff_check() {
    log "[INFO] Running clang-tidy-diff check (base: $BASE_BRANCH)..."

    [[ -n "$COMPILE_DB" ]] && log "[INFO] Using compile_commands.json: $COMPILE_DB"
    [[ -z "$COMPILE_DB" ]] && log "[WARNING] compile_commands.json not found, using manual flags"

    local tidy_output=""
    if [[ -n "$COMPILE_DB" ]]; then
        tidy_output=$(git diff -U0 "$BASE_BRANCH" HEAD | clang-tidy-diff -p1 -clang-tidy-binary clang-tidy -path "$(dirname "$COMPILE_DB")" 2>&1) || true
    else
        local tidy_args=(-p1 -clang-tidy-binary clang-tidy -extra-arg-before="-x" -extra-arg-before="c++")
        for flag in $MANUAL_COMPILE_FLAGS; do
            tidy_args+=(-extra-arg-before="$flag")
        done
        tidy_output=$(git diff -U0 "$BASE_BRANCH" HEAD | clang-tidy-diff "${tidy_args[@]}" 2>&1) || true
    fi

    if [[ -n "$tidy_output" ]]; then
        local manual_flags_flag=0
        [[ -z "$COMPILE_DB" ]] && manual_flags_flag=1
        # Pass the ignore list to awk via NUL-separated -v? awk -v can't take arrays;
        # serialize the list into one variable separated by RS-safe newlines.
        local ignore_blob
        ignore_blob=$(printf '%s\n' "${TIDY_IGNORE_LIST[@]}")

        set +e
        local filtered_output=$(echo "$tidy_output" | awk -v manual_flags="$manual_flags_flag" -v ignore_list="$ignore_blob" '
        function flush_block() {
            if (!drop && block_len > 0) {
                for (i = 0; i < block_len; i++) print block[i]
            }
            block_len = 0
            drop = 0
            check_name = ""
            block_path = ""
            block_symbol = ""
        }
        function should_drop_path(p) {
            return (p ~ /mlir\// || p ~ /llvm\// || p ~ /\.inc/)
        }
        function should_drop_clang_diagnostic(c) {
            return (manual_flags == 1 && c ~ /^clang-diagnostic-/)
        }
        function should_drop_whitelisted(p, sym, c,    n, i, parts, suf, wsym, wcheck) {
            if (ignore_list == "") return 0
            n = split(ignore_list, lines, "\n")
            for (i = 1; i <= n; i++) {
                if (lines[i] == "") continue
                split(lines[i], parts, "|")
                suf = parts[1]
                wsym = parts[2]
                wcheck = parts[3]
                if (suf == "" || wsym == "" || wcheck == "") continue
                if (index(p, suf) == 0) continue
                if (sym != wsym) continue
                if (c != wcheck) continue
                return 1
            }
            return 0
        }
        BEGIN { block_len = 0; drop = 0 }
        /^[^ ]+:[0-9]+:[0-9]+:.*(warning:|error:|note:)/ {
            flush_block()
            block_path = $0
            sub(/:[0-9]+:[0-9]+:.*/, "", block_path)
            check_name = ""
            block_symbol = ""
            if (match($0, /\[[a-zA-Z0-9_.-]+\]$/)) {
                check_name = substr($0, RSTART + 1, RLENGTH - 2)
            }
            if (match($0, /'\''[^'\'']+'\''/)) {
                block_symbol = substr($0, RSTART + 1, RLENGTH - 2)
            }
            if (should_drop_path(block_path)) {
                drop = 1
            } else if (should_drop_clang_diagnostic(check_name)) {
                drop = 1
            } else if (should_drop_whitelisted(block_path, block_symbol, check_name)) {
                drop = 1
            }
            block[block_len++] = $0
            next
        }
        /^[[:space:]]/ {
            if (block_len > 0) { block[block_len++] = $0; next }
            next
        }
        { flush_block() }
        END { flush_block() }')
        set -e

        if [[ -n "$filtered_output" ]]; then
            TIDY_ERROR_COUNT=$(echo "$filtered_output" | grep -c "error:" | xargs || echo 0)
            TIDY_WARNING_COUNT=$(echo "$filtered_output" | grep -c "warning:" | xargs || echo 0)

            log "[TIDY ISSUE] Found static analysis issues"
            echo "$filtered_output"

            log "[HINT] Reproduce: git diff -U0 $BASE_BRANCH HEAD | clang-tidy-diff -p1 -clang-tidy-binary clang-tidy -extra-arg-before='-x' -extra-arg-before='c++' ${MANUAL_COMPILE_FLAGS//-I/-extra-arg-before=-I}"
        fi
    fi

    [[ "$TIDY_ERROR_COUNT" -gt 0 ]] && log "[ERROR] $TIDY_ERROR_COUNT errors"
    [[ "$TIDY_WARNING_COUNT" -gt 0 ]] && log "[WARNING] $TIDY_WARNING_COUNT warnings"
    return 0
}

# ============================================================================
# Main Entry
# ============================================================================

print_summary() {
    echo ""
    echo "----------------------------------------------------------------"
    echo "Static Check Summary"
    echo "----------------------------------------------------------------"
    echo "clang-tidy-diff:   $TIDY_ERROR_COUNT error(s), $TIDY_WARNING_COUNT warning(s)"
    echo "----------------------------------------------------------------"

    if [[ $TIDY_ERROR_COUNT -gt 0 || $TIDY_WARNING_COUNT -gt 0 ]]; then
        echo "Status: FAILED"
        return 1
    fi
    echo "Status: PASSED"
}

main() {
    log "[INFO] Checking changed code between $BASE_BRANCH and HEAD"

    local cpp_files=$(git diff --name-only "$BASE_BRANCH" HEAD | grep -E '\.cpp$|\.h$')

    if [[ -z "$cpp_files" ]]; then
        log "[INFO] No C/C++ files changed"
        print_summary
        exit 0
    fi

    log "[INFO] Changed C/C++ files: $cpp_files"

    check_and_install_tools
    setup_compile_flags

    run_clang_tidy_diff_check

    print_summary
    exit $?
}

main
