#!/usr/bin/env bash

# Copyright (c) 2026 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#   bash scripts/build-bench.sh              # fresh build timing
#   bash scripts/build-bench.sh --incremental  # incremental build timing

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/bench-$(date +%s)-$$"
INCREMENTAL=false

[[ $# -gt 0 && "$1" == "--incremental" ]] && INCREMENTAL=true

measure_build() {
  local label="$1"; shift
  local build_dir="$1"; shift
  local no_clean=false
  local start_time end_time elapsed log_file

  [[ $# -gt 0 && "$1" == "--no-clean" ]] && { no_clean=true; shift; }

  if ! $no_clean; then
    rm -rf "$build_dir"
    mkdir -p "$build_dir"
  fi

  log_file="${build_dir}/build.log"
  start_time=$(date +%s.%N)
  PYASC_BUILD_TIMING=1 PYASC_SETUP_BUILD_DIR="$build_dir" \
    python3 -m pip install --no-build-isolation -e "$REPO_ROOT" > "$log_file" 2>&1
  end_time=$(date +%s.%N)

  elapsed=$(awk "BEGIN {print $end_time - $start_time}")
  echo "${label}: ${elapsed}s (log: ${log_file})"
}

if $INCREMENTAL; then
  measure_build "full"         "$BUILD_DIR"
  measure_build "incremental"  "$BUILD_DIR" --no-clean
else
  measure_build "fresh" "$BUILD_DIR"
fi

rm -rf "$BUILD_DIR"
echo "Done."
