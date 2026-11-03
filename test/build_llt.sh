#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -e
CURRENT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
BUILD_DIR=${CURRENT_DIR}/build
OUTPUT_DIR=${CURRENT_DIR}/output
TEST_PATH=${CURRENT_DIR}/python
UT_PATH=${CURRENT_DIR}/../python/test/unit
CPU_NUM="$(nproc)"
JOB_NUM="-j$((CPU_NUM < 16 ? CPU_NUM : 16))"  # Limit parallel build jobs to 16
ASAN="false"
COV="false"
PYASC_SETUP_CCACHE="ON"
PYASC_SETUP_CLANG_LLD="OFF"

# PR修改文件列表路径，通过 -f|--filelist 参数传入
# 用于分析修改内容，决定测试触发策略
PR_FILELIST=""

# C++测试目标模块
# 值为 "all" 时执行全量Lit测试
# 值为具体模块名（Adv/Basic/Core/External/Fwk）时执行该模块精准测试
CPP_TEST_TARGET="all"

# Python测试目标模块
# 值为 "all" 时执行全量pytest测试
# 值为具体模块名（language/adv等）时执行该模块精准测试
PYTHON_TEST_TARGET="all"

# 是否需要执行C++ Lit测试
# 由 analyze_pr_filelist() 函数根据PR文件分析结果决定
NEED_CPP_TEST="false"

# 是否需要执行Python UT测试
# 由 analyze_pr_filelist() 函数根据PR文件分析结果决定
NEED_PYTHON_TEST="false"

# 测试跳过白名单配置
# 匹配以下类型的文件修改时不触发测试：
# - 文档类：docs/, *.md, *.txt, *.rst
# - 配置类：.github/, .gitignore, .clang-format, .gitattributes
# - 其他：LICENSE, README, CHANGELOG, images/, assets/
NO_TEST_WHITELIST=(
    "docs/"
    "LICENSE"
    "README"
    ".github/"
    ".gitignore"
    "*.md"
    "*.txt"
    "*.rst"
    "images/"
    "assets/"
    ".clang-format"
    ".gitattributes"
    "CHANGELOG"
    "CONTRIBUTING"
)

# 已知C++模块列表
# 对应 lib/Target/AscendC/{模块名}/ 目录下的源码
# 精准匹配后仅触发对应模块的Lit测试
KNOWN_CPP_MODULES=("Adv" "Basic" "Core" "External" "Fwk")

# 已知Python模块列表
# 对应 python/asc/{模块名}/ 目录下的源码
# 精准匹配后仅触发对应模块的pytest测试
KNOWN_PYTHON_MODULES=("language/adv" "language/basic" "language/core" "language/fwk" "lib/host" "lib/runtime" "codegen" "runtime")

# 全量测试触发路径
# 修改以下路径下的文件会触发全量C++和Python测试：
# - 核心：lib/Dialect/, lib/TableGen/, include/, bin/
# - 构建配置：CMakeLists.txt, cmake/
FULL_TEST_PATHS=(
    "lib/Dialect/Asc/IR/"
    "lib/Dialect/Asc/Transforms/"
    "lib/Dialect/EmitAsc/"
    "lib/TableGen/"
    "include/"
    "bin/"
    "CMakeLists.txt"
    "cmake/"
)

PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
CUSTOM_OPTION="-DASCIR_LLT_TEST=ON -DCMAKE_INSTALL_PREFIX=${OUTPUT_DIR} -Dpybind11_DIR=${PYBIND11_DIR}"

function log() {
    local current_time=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[$current_time] "$1
}

# 检查文件是否在白名单中
# 参数：$1 - 文件路径
# 返回：0 - 在白名单中（跳过测试）
#       1 - 不在白名单中（继续分析）
function is_in_whitelist() {
    local file="$1"
    for pattern in "${NO_TEST_WHITELIST[@]}"; do
        if [[ "$file" == ${pattern}* ]] || [[ "$file" == ${pattern} ]]; then
            return 0
        fi
    done
    return 1
}

# 分析PR文件列表，决定测试触发策略
#
# 分析流程：
# 1. 白名单检查：匹配白名单模式的文件跳过
# 2. 全量触发检查：匹配FULL_TEST_PATHS的文件触发全量测试
# 3. 精准匹配检查：
#    - C++：lib/Target/AscendC/{模块}/ 匹配到KNOWN_CPP_MODULES
#    - Python：python/asc/{模块}/ 匹配到KNOWN_PYTHON_MODULES
#    - 单模块精准测试，多模块触发全量
# 4. 公共文件检查：lib/Target/AscendC/*.cpp 非子目录文件触发全量
# 5. 未知源码兜底：未知路径触发全量测试（安全策略）
#
# 输出变量：
# - NEED_CPP_TEST：是否需要C++测试
# - CPP_TEST_TARGET：C++测试目标（all/模块名）
# - NEED_PYTHON_TEST：是否需要Python测试
# - PYTHON_TEST_TARGET：Python测试目标（all/模块名）
function analyze_pr_filelist() {
    local file_list="${PR_FILELIST}"

    CPP_TEST_TARGET="all"
    PYTHON_TEST_TARGET="all"
    NEED_CPP_TEST="false"
    NEED_PYTHON_TEST="false"
    local HAS_UNKNOWN_SOURCE="false"

    if [[ -z "${file_list}" ]] || [[ ! -f "${file_list}" ]]; then
        NEED_CPP_TEST="true"
        NEED_PYTHON_TEST="true"
        log "Info: No PR file list provided, running full tests"
        return
    fi

    local cpp_module_hits=()
    local python_module_hits=()
    local whitelist_hits=0
    local source_hits=0

    while IFS= read -r file || [[ -n "$file" ]]; do
        [[ -z "$file" ]] && continue

        if is_in_whitelist "$file"; then
            whitelist_hits=$((whitelist_hits + 1))
            continue
        fi

        for full_path in "${FULL_TEST_PATHS[@]}"; do
            if [[ "$file" == ${full_path}* ]]; then
                log "Info: Core path modified: $file -> full tests required"
                NEED_CPP_TEST="true"
                NEED_PYTHON_TEST="true"
                CPP_TEST_TARGET="all"
                PYTHON_TEST_TARGET="all"
                return
            fi
        done

        # 步骤1: 先精准匹配模块子目录
        local matched_cpp="false"
        local cpp_matched_module=""
        for module in "${KNOWN_CPP_MODULES[@]}"; do
            if [[ "$file" == lib/Target/AscendC/${module}/* ]]; then
                NEED_CPP_TEST="true"
                source_hits=$((source_hits + 1))
                cpp_module_hits+=("$module")
                cpp_matched_module="$module"
                matched_cpp="true"
                break
            fi
        done

        # 步骤2: lib/Target/AscendC 目录下的公共文件（非子目录）触发全量
        if [[ "$file" == lib/Target/AscendC/* ]] && [[ "$matched_cpp" == "false" ]]; then
            if [[ "$file" != lib/Target/AscendC/*/* ]]; then
                log "Info: Public source file in lib/Target/AscendC modified: $file -> full tests required"
                NEED_CPP_TEST="true"
                CPP_TEST_TARGET="all"
                PYTHON_TEST_TARGET="all"
                NEED_PYTHON_TEST="true"
                return
            else
                HAS_UNKNOWN_SOURCE="true"
                log "Warning: Unknown C++ source path: $file"
            fi
        fi

        local matched_python="false"
        local py_matched_module=""
        if [[ "$file" == python/asc/* ]]; then
            NEED_PYTHON_TEST="true"
            source_hits=$((source_hits + 1))
            matched_python="true"

            for module in "${KNOWN_PYTHON_MODULES[@]}"; do
                if [[ "$file" == python/asc/${module}* ]]; then
                    python_module_hits+=("$module")
                    py_matched_module="$module"
                    break
                fi
            done

            if [[ -z "${py_matched_module}" ]]; then
                HAS_UNKNOWN_SOURCE="true"
                log "Warning: Unknown Python source path: $file"
            fi
        fi

        if [[ "$file" == lib/* ]] || [[ "$file" == python/* ]]; then
            if [[ "${matched_cpp}" == "false" ]] && [[ "${matched_python}" == "false" ]]; then
                source_hits=$((source_hits + 1))
                HAS_UNKNOWN_SOURCE="true"
                log "Warning: Unknown source path: $file"
            fi
        fi

    done < "$file_list"

    log "Info: File analysis: whitelist=${whitelist_hits}, source=${source_hits}, unknown_source=${HAS_UNKNOWN_SOURCE}"

    if [[ "${HAS_UNKNOWN_SOURCE}" == "true" ]]; then
        log "Warning: Unknown source directory modified, triggering full tests for safety"
        CPP_TEST_TARGET="all"
        PYTHON_TEST_TARGET="all"
        NEED_CPP_TEST="true"
        NEED_PYTHON_TEST="true"
        return
    fi

    if [[ ${#cpp_module_hits[@]} -gt 0 ]]; then
        local unique_cpp=$(printf '%s\n' "${cpp_module_hits[@]}" | sort -u | tr '\n' ' ' | sed 's/ $//')
        local cpp_count=$(echo "$unique_cpp" | wc -w)
        if [[ "$cpp_count" -eq 1 ]]; then
            CPP_TEST_TARGET="$unique_cpp"
            log "Info: C++ precise test target: ${CPP_TEST_TARGET}"
        elif [[ "$cpp_count" -gt 1 ]]; then
            CPP_TEST_TARGET="all"
            log "Info: Multiple C++ modules modified (${unique_cpp}), full test"
        fi
    fi

    if [[ ${#python_module_hits[@]} -gt 0 ]]; then
        local unique_python=$(printf '%s\n' "${python_module_hits[@]}" | sort -u | tr '\n' ' ' | sed 's/ $//')
        local python_count=$(echo "$unique_python" | wc -w)
        if [[ "$python_count" -eq 1 ]]; then
            PYTHON_TEST_TARGET="$unique_python"
            log "Info: Python precise test target: ${PYTHON_TEST_TARGET}"
        elif [[ "$python_count" -gt 1 ]]; then
            PYTHON_TEST_TARGET="all"
            log "Info: Multiple Python modules modified (${unique_python}), full test"
        fi
    fi

    if [[ "${NEED_CPP_TEST}" == "false" ]] && [[ "${NEED_PYTHON_TEST}" == "false" ]]; then
        log "Info: No source code changes detected, tests may be skipped"
    fi
}

function clean()
{
    if [[ -n "${BUILD_DIR}" ]];then
        rm -rf ${BUILD_DIR}
    fi

    if [[ -z "${TEST}" ]];then
        if [ -n "${OUTPUT_DIR}" ];then
            rm -rf ${OUTPUT_DIR}
        fi
    fi

    mkdir -p ${BUILD_DIR} ${OUTPUT_DIR}
}

function cmake_config()
{
    local extra_option="$1"
    log "Info: cmake config ${CUSTOM_OPTION} ${extra_option} ."
    cmake ../..  ${CUSTOM_OPTION} ${extra_option}
}

# 执行Python单元测试
#
# 测试策略：
# - PYTHON_TEST_TARGET=all：执行 $UT_PATH 全量pytest测试
# - PYTHON_TEST_TARGET=模块名：执行 $UT_PATH/{模块名}/ 精准pytest测试
# - 特殊处理：全量测试按 host 和非host 分组，lib/host 精准测试仅执行串行 host 测试
#
# 覆盖率支持：
# - COV=true：使用 coverage 模块生成覆盖率数据到 ${CURRENT_DIR}/cov_py/
function run_python_ut()
{
    export PATH="$LLVM_INSTALL_PATH/bin:$PATH"

    local test_path="$UT_PATH"

    if [[ "${PYTHON_TEST_TARGET}" != "all" ]]; then
        test_path="$UT_PATH/${PYTHON_TEST_TARGET}"
        if [[ -d "${test_path}" ]]; then
            log "Info: Running precise Python UT: ${test_path}"
        else
            log "Warning: Test path ${test_path} not found, running full tests"
            test_path="$UT_PATH"
        fi
    else
        log "Info: Running full Python UT: ${test_path}"
    fi

    if [[ "${COV}" == "true" ]];then
        COV_DIR=${CURRENT_DIR}/cov_py
        SRC_DIR=$(python3 -c "import pkg_resources; print(pkg_resources.resource_filename('asc', ''))")
        log "Info: source directory is ${SRC_DIR}"

        python3 -m coverage run --source=$SRC_DIR --data-file="${COV_DIR}/.coverage" -m pytest -v ${test_path}

        log "Info: coverage data directory is ${COV_DIR}"

        cd "${COV_DIR}"
        coverage report
        coverage html
    else
        if [[ "${PYTHON_TEST_TARGET}" == "lib/host" ]] && [[ "${test_path}" == "$UT_PATH/lib/host" ]]; then
            python3 -m pytest -v "${test_path}" -n 1 -k "host"
        elif [[ "${PYTHON_TEST_TARGET}" != "all" ]] && [[ "${PYTHON_TEST_TARGET}" != "lib/host" ]]; then
            python3 -m pytest -v ${test_path} -n auto
        else
            python3 -m pytest -v ${test_path} -n auto -k "not host"
            python3 -m pytest -v ${test_path} -n 1 -k "host"
        fi
    fi
}

# 执行C++ Lit测试
#
# 测试策略：
# - CPP_TEST_TARGET=all：执行 make check-ascir 全量测试
# - CPP_TEST_TARGET=模块名：
#   1. 仅构建必要工具（ascir-opt, ascir-translate）
#   2. 运行对应模块的Lit测试文件（{模块名}.mlir 或 {模块名}/ 目录）
#   3. 若找不到对应测试文件，回退到全量测试
#
# 覆盖率支持：
# - COV=true：生成覆盖率数据到 ${BUILD_DIR}/coverage.info
function run_check_ascir()
{
    clean
    cd ${BUILD_DIR}
    if [[ -z "${LIT_INSTALL_PATH}" ]] || [[ ! -e "${LIT_INSTALL_PATH}" ]];then
        cmake_config "-DLLVM_PREFIX_PATH=${LLVM_INSTALL_PATH} -DLLVM_EXTERNAL_LIT=${LLVM_INSTALL_PATH}/bin/llvm-lit"
    else
        cmake_config "-DLLVM_PREFIX_PATH=${LLVM_INSTALL_PATH} -DLLVM_EXTERNAL_LIT=${LIT_INSTALL_PATH}/bin/lit"
    fi

    if [[ "${COV}" == "true" ]]; then
        export LLVM_PROFILE_FILE="${BUILD_DIR}/%p.profraw"
        log "Info: LLVM_PROFILE_FILE set to ${BUILD_DIR}/%p.profraw"
    fi

    if [[ "${CPP_TEST_TARGET}" == "all" ]]; then
        log "Info: Running full C++ Lit tests"
        make check-ascir ${JOB_NUM}
    else
        log "Info: Running precise C++ Lit tests for: ${CPP_TEST_TARGET}"
        make ${JOB_NUM} ascir-opt ascir-translate

        local test_base="${CURRENT_DIR}/../test/Target/AscendC"
        local test_file="${test_base}/${CPP_TEST_TARGET}.mlir"
        local test_dir="${test_base}/${CPP_TEST_TARGET}"

        if [[ -f "${test_file}" ]]; then
            lit -v "${test_file}" --param ascir_tools_dir=${BUILD_DIR}/bin
        elif [[ -d "${test_dir}" ]]; then
            lit -v "${test_dir}" --param ascir_tools_dir=${BUILD_DIR}/bin
        else
            log "Warning: Test path not found for ${CPP_TEST_TARGET}, running full tests"
            make check-ascir ${JOB_NUM}
        fi
    fi

    cd ${CURRENT_DIR}

    if [[ "${COV}" == "true" ]];then
        cov_file=${BUILD_DIR}/coverage.info
        generate_coverage "${BUILD_DIR}" "${cov_file}"
        filter_coverage   "${cov_file}" "${cov_file}_filtered"
        generate_html     "${cov_file}_filtered" "${CURRENT_DIR}/cov_ascir"
    fi
}

# check clang/lld availability
check_clang_available() {
  if [[ "${PYASC_SETUP_CLANG_LLD}" == "ON" ]]; then
    if ! command -v clang >/dev/null 2>&1; then
      log "Error: clang not found, please install clang or run without --clang"
      exit 1
    fi
    if ! command -v clang++ >/dev/null 2>&1; then
      log "Error: clang++ not found, please install clang or run without --clang"
      exit 1
    fi
    if ! command -v lld >/dev/null 2>&1 && ! command -v ld.lld >/dev/null 2>&1; then
      log "Error: lld not found, please install lld or run without --clang"
      exit 1
    fi
    log "Info: clang/lld detected and available"
  fi
}

# detect compiler type
detect_compiler() {
  if [[ "${PYASC_SETUP_CLANG_LLD}" == "ON" ]]; then
    echo "clang"
  else
    echo "gcc"
  fi
}

# using lcov to generate coverage for cpp files
generate_coverage() {
  local _source_dir="$1"
  local _coverage_file="$2"

  if [[ -z "${_source_dir}" ]]; then
    log "Info: directory required to find coverage files"
    exit 1
  fi

  if [[ ! -d "${_source_dir}" ]]; then
    log "Info: directory is not exist, please check ${_source_dir}"
    exit 1
  fi

  if [[ -z "${_coverage_file}" ]]; then
    _coverage_file="coverage.info"
    log "Info: using default file name to generate coverage"
  fi

  local _path_to_gen="$(dirname ${_coverage_file})"
  if [[ ! -d "${_path_to_gen}" ]]; then
    mkdir -p "${_path_to_gen}"
  fi

  local compiler=$(detect_compiler)
  log "Info: detected compiler: ${compiler}"

  if [[ "${compiler}" == "clang" ]]; then
    generate_llvm_coverage "${_source_dir}" "${_coverage_file}"
  elif [[ "${compiler}" == "gcc" ]]; then
    generate_gcc_coverage "${_source_dir}" "${_coverage_file}"
  else
    log "Info: unknown compiler, trying gcc coverage generation"
    generate_gcc_coverage "${_source_dir}" "${_coverage_file}"
  fi
}

# generate coverage for gcc compiler
generate_gcc_coverage() {
  local _source_dir="$1"
  local _coverage_file="$2"

  \which lcov >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    log "Info: lcov is required to generate coverage data, please install"
    exit 1
  fi

  local _path_to_gen="$(dirname ${_coverage_file})"
  if [[ ! -d "${_path_to_gen}" ]]; then
    mkdir -p "${_path_to_gen}"
  fi

  local lcov_ignore_errors=""
  local lcov_version=$(lcov --version 2>&1 | grep -oP 'version \K[0-9]+' | head -1)
  if [[ "${lcov_version}" -ge 2 ]]; then
    lcov_ignore_errors="mismatch,empty,inconsistent,negative,source,unused"
  else
    lcov_ignore_errors="source,graph"
  fi
  log "Info: lcov version ${lcov_version}, using ignore-errors: ${lcov_ignore_errors}"

  lcov -c -d "${_source_dir}" -o "${_coverage_file}" --rc geninfo_unexecuted_blocks=1 --ignore-errors ${lcov_ignore_errors}
  lcov -r "${_coverage_file}" "/home/jenkins/Ascend/ascend-toolkit/latest/*" -o "${_coverage_file}" --ignore-errors unused
  log "Info: generated coverage file ${_coverage_file}"
}

# generate coverage for clang compiler
generate_llvm_coverage() {
  local _source_dir="$1"
  local _coverage_file="$2"

  local llvm_profdata=""
  local llvm_cov=""

  if [[ "${PYASC_SETUP_CLANG_LLD}" == "ON" ]] && [[ -n "${CLANG_CMD}" ]]; then
    local clang_version=""
    if [[ "${CLANG_CMD}" == "clang" ]]; then
      clang_version=$(clang --version 2>&1 | grep -oP 'version \K[0-9]+' | head -1)
    else
      clang_version=$(echo "${CLANG_CMD}" | grep -oP 'clang-\K[0-9]+')
    fi
    log "Info: detected clang version: ${clang_version}"

    if [[ -n "${clang_version}" ]] && command -v llvm-profdata-${clang_version} >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-${clang_version}"
      llvm_cov="llvm-cov-${clang_version}"
      log "Info: using version-matched tools: ${llvm_profdata}"
    fi
  fi

  if [[ -z "${llvm_profdata}" ]]; then
    if command -v llvm-profdata >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata"
      llvm_cov="llvm-cov"
    elif command -v llvm-profdata-19 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-19"
      llvm_cov="llvm-cov-19"
    elif command -v llvm-profdata-18 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-18"
      llvm_cov="llvm-cov-18"
    elif command -v llvm-profdata-17 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-17"
      llvm_cov="llvm-cov-17"
    elif command -v llvm-profdata-16 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-16"
      llvm_cov="llvm-cov-16"
    elif command -v llvm-profdata-15 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-15"
      llvm_cov="llvm-cov-15"
    elif command -v llvm-profdata-14 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-14"
      llvm_cov="llvm-cov-14"
    elif command -v llvm-profdata-13 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-13"
      llvm_cov="llvm-cov-13"
    elif command -v llvm-profdata-12 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-12"
      llvm_cov="llvm-cov-12"
    elif command -v llvm-profdata-11 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-11"
      llvm_cov="llvm-cov-11"
    elif command -v llvm-profdata-10 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-10"
      llvm_cov="llvm-cov-10"
    else
      log "Error: llvm-profdata not found, please install llvm package"
      exit 1
    fi
    log "Info: using system tools: ${llvm_profdata}"
  fi

  log "Info: using ${llvm_profdata} and ${llvm_cov} for coverage processing"

  local profdata_file="${_source_dir}/coverage.profdata"
  local profraw_files=$(find "${_source_dir}" -name "*.profraw" 2>/dev/null)

  if [[ -z "${profraw_files}" ]]; then
    log "Info: no .profraw files found in ${_source_dir}"
    exit 1
  fi

  log "Info: found .profraw files"

  ${llvm_profdata} merge -sparse ${profraw_files} -o "${profdata_file}"
  if [[ $? -ne 0 ]]; then
    log "Info: failed to merge profdata files"
    exit 1
  fi

  log "Info: generated profdata file ${profdata_file}"

  local temp_coverage="${_source_dir}/coverage_temp.info"
  rm -f "${temp_coverage}"

  for binary in "${_source_dir}/bin/ascir-opt" "${_source_dir}/bin/ascir-translate"; do
    if [[ -f "${binary}" ]]; then
      log "Info: exporting coverage for ${binary}"
      ${llvm_cov} export "${binary}" -instr-profile="${profdata_file}" -format=lcov >> "${temp_coverage}"
    fi
  done

  if [[ ! -f "${temp_coverage}" ]]; then
    log "Info: no coverage data exported"
    exit 1
  fi

  mv "${temp_coverage}" "${_coverage_file}"

  log "Info: generated coverage file ${_coverage_file}"
}

# filter out some unused directories or files
filter_coverage() {
  local _coverage_file="$1"
  local _filtered_file="$2"

  if [[ ! -f "${_coverage_file}" ]]; then
    log "Info: coverage data file required"
    exit 1
  fi

  \which lcov >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    log "Info: lcov is required to generate coverage data, please install"
    exit 1
  fi

  local lcov_version=$(lcov --version 2>&1 | grep -oP 'version \K[0-9]+' | head -1)
  local lcov_filter_ignore=""
  if [[ "${lcov_version}" -ge 2 ]]; then
    lcov_filter_ignore="--ignore-errors unused"
  fi

  lcov --remove "${_coverage_file}" \
       '/usr/include/*' '/usr/local/include/*' \
       '*/llvm-install/*' '*/llvm/*' '*/mlir/*' \
       "${BUILD_DIR}/*" \
       -o "${_filtered_file}" ${lcov_filter_ignore}

  log "Info: Coverage statistics:"
  lcov --summary "${_filtered_file}" 2>&1 | grep -E "lines|functions"
}

# generate html report
generate_html() {
  local _filtered_file="$1"
  local _out_path="$2"

  \which genhtml >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    log "Info: genhtml is required to generate coverage html report, please install"
    exit 1
  fi

  local _path_to_gen="$(dirname ${_out_path})"
  if [[ ! -d "${_out_path}" ]]; then
    mkdir -p "${_out_path}"
  fi
  genhtml "${_filtered_file}" -o "${_out_path}"
}

# 测试执行入口函数
#
# 执行逻辑：
# 1. 调用 analyze_pr_filelist() 分析PR文件
# 2. 根据TEST参数决定执行流程：
#    - TEST=lit：仅执行C++测试，NEED_CPP_TEST=false时 exit 200
#    - TEST=python_ut：仅执行Python测试，NEED_PYTHON_TEST=false时 exit 200
#    - TEST=all：执行全部测试，跳过部分不exit
#
# Exit Code：
# - 0：测试正常执行完成
# - 200：测试跳过（非错误状态）
# - 1：测试执行失败
function build_test() {
    analyze_pr_filelist

    log "Info: Test decision: CPP=${NEED_CPP_TEST}(${CPP_TEST_TARGET}), Python=${NEED_PYTHON_TEST}(${PYTHON_TEST_TARGET})"

    if [[ "${TEST}" == "lit" ]]; then
        if [[ "${NEED_CPP_TEST}" == "true" ]]; then
            run_check_ascir
        else
            log "Info: Skip C++ Lit test - no relevant source changes"
            exit 200
        fi
        log "Info: test check-ascir completed."
    elif [[ "${TEST}" == "python_ut" ]]; then
        if [[ "${NEED_PYTHON_TEST}" == "true" ]]; then
            run_python_ut
        else
            log "Info: Skip Python UT test - no relevant source changes"
            exit 200
        fi
        log "Info: test run_python_ut completed."
    else
        if [[ "${NEED_CPP_TEST}" == "true" ]]; then
            run_check_ascir
        else
            log "Info: Skip C++ Lit test - no relevant source changes"
        fi
        if [[ "${NEED_PYTHON_TEST}" == "true" ]]; then
            run_python_ut
        else
            log "Info: Skip Python UT test - no relevant source changes"
        fi
        log "Info: test all completed."
    fi
}


TEST="all"
while [[ $# -gt 0 ]]; do
    case $1 in
    -f|--filelist)
        PR_FILELIST="$2"
        shift 2
        ;;
    --check-ascir)
        TEST="lit"
        shift
        ;;
    --clang)
        PYASC_SETUP_CLANG_LLD="ON"
        shift
        ;;
    --llvm_install_path)
        LLVM_INSTALL_PATH="$2"
        shift 2
        ;;
    --lit_install_path)
        LIT_INSTALL_PATH="$2"
        shift 2
        ;;
    --run_python_ut)
        TEST="python_ut"
        shift
        ;;
    --asan)
        ASAN="true"
        shift
        ;;
    --cov)
        COV="true"
        shift
        ;;
    *)
        break
        ;;
    esac
done

if [[ -n "${PR_FILELIST}" ]]; then
    if [[ -f "${PR_FILELIST}" ]]; then
        log "Info: PR file list from ${PR_FILELIST}:"
        cat "${PR_FILELIST}"
        log "Info: Analyzing PR file list for precise testing..."
    else
        log "Warning: PR file list file ${PR_FILELIST} not found, running full tests"
    fi
fi

if [[ -z "${LLVM_INSTALL_PATH}" ]] || [[ ! -e "${LLVM_INSTALL_PATH}" ]];then
    log "Error: --llvm_install_path need to be set or LLVM_INSTALL_PATH: ${LLVM_INSTALL_PATH} is not exsit."
    exit 1
fi

if [[ -z "${LIT_INSTALL_PATH}" ]] || [[ ! -e "${LIT_INSTALL_PATH}" ]];then
    if [[ "${TEST}" != "python_ut" ]];then
        log "Warning: --lit_install_path need to be set, otherwise it will use default path: ${LLVM_INSTALL_PATH},\
            please make sure lit exsit at default path!"
    fi
fi

if [[ "${ASAN}" == "true" ]];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DASCIR_ASAN=true"
fi

if [[ "${COV}" == "true" ]];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DASCIR_COVERAGE=true"
fi

check_clang_available

if [[ "${PYASC_SETUP_CCACHE}" == "ON" ]];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DASCIR_CCACHE=ON"
fi

if [[ "${PYASC_SETUP_CLANG_LLD}" == "ON" ]];then
    CUSTOM_OPTION="${CUSTOM_OPTION}
    -DCMAKE_C_COMPILER=clang
    -DCMAKE_CXX_COMPILER=clang++
    -DCMAKE_LINKER=lld
    -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld
    -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld
    -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld"
fi
build_test
