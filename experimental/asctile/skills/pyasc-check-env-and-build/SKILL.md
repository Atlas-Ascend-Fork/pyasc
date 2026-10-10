---
name: pyasc-check-env-and-build
description: Environment setup and build pipeline for the PyAsc project. Checks LLVM, determines build context (developer vs user), verifies dependencies, and builds the project. Trigger when user asks to build, install, or set up the PyAsc development environment.
---

# PyAsc Build

## When to Use

Use this skill when:
- A user asks to build or install the PyAsc project
- A developer needs to set up the build environment from scratch
- Environment variables need to be configured for compilation
- The user wants to verify that all build prerequisites are met
- A new contributor needs guidance through the full setup and build process

## Agent Pipeline

### Phase 1: Check LLVM Installation

**Goal**: Verify that a pre-built LLVM repository exists and `LLVM_INSTALL_PREFIX` is set.

**Steps**:

1. **Check if `LLVM_INSTALL_PREFIX` environment variable is set**
   ```bash
   echo $LLVM_INSTALL_PREFIX
   ```

2. **If not set, search standard locations**
   - Check parent directory (most common): `ls -d ../llvm-*`
   - Check other locations: `/usr/local/llvm-*`, `~/llvm-*`, `$HOME/.local/llvm-*`
   - Ask user to provide the full path to LLVM installation
   - If found: Suggest setting `LLVM_INSTALL_PREFIX` and add `export LLVM_INSTALL_PREFIX=<path>` to the current build session

3. **Verify LLVM libraries and CMake configs exist** (compilers are installed system-wide via `apt install build-essential ccache clang lld`)
   ```bash
   ls -la $LLVM_INSTALL_PREFIX/lib/cmake/mlir/MLIRConfig.cmake
   ls -la $LLVM_INSTALL_PREFIX/lib/cmake/llvm/LLVMConfig.cmake
   ls -d $LLVM_INSTALL_PREFIX/include/mlir
   ls -d $LLVM_INSTALL_PREFIX/include/llvm
   ls -la $LLVM_INSTALL_PREFIX/bin/mlir-tblgen
   ls -la $LLVM_INSTALL_PREFIX/bin/FileCheck*
   ```

4. **Check versions**
   ```bash
   $LLVM_INSTALL_PREFIX/bin/llvm-config --version
   ```

**If LLVM not found**:

Report the error to the user. Download links can be found in @docs/installation/build-from-source.rst.

Ask the user where to download LLVM (suggest home directory `~/llvm` as default location).

**Failure handling**:
- LLVM not found: Provide download links above
- Missing CMake configs or headers: "LLVM installation appears incomplete. Missing: {path}"
- Permission errors: Suggest `chmod +x $LLVM_INSTALL_PREFIX/bin/*`

---

### Phase 2: Determine Build Context

**Goal**: Determine if the user is a developer (likely needs Debug mode) or just building for usage (Release mode). This affects which environment variables are set.

**Step 1 (MANDATORY)**: Ask the user which build configuration they need.

Ask exactly this question:
"Which build configuration do you need?
- **Developer build** (Debug): Includes devtools (ascir-opt, ascir-lsp, ascir-translate), ccache, clang/lld. Recommended for contributing to pyasc.
- **Standard build** (Release): Minimal build for running kernels. Debug tools (ascir-opt, ascir-lsp, ascir-translate) will not be available."

**DO NOT** proceed to Step 2 until you receive an answer. **DO NOT** infer the answer from context clues (source code presence, .venv, AGENTS.md, etc.).

**Step 2**: Based on user's answer, set environment variables:

- If user chooses Developer build: **Developer context** (Debug build)
- If user chooses Standard build: **User context** (Release build)
- If user's response is still ambiguous: Ask for clarification

**Developer context** environment variables:

```bash
export LLVM_INSTALL_PREFIX=/path/to/llvm          # required
export PYASC_SETUP_CONFIG=Debug                    # Debug for development
export PYASC_SETUP_DEVTOOLS=1                      # build ascir-opt, ascir-lsp, ascir-translate
export PYASC_SETUP_CCACHE=1                        # faster rebuilds
export PYASC_SETUP_CLANG_LLD=1                     # use clang/lld
export PYASC_SETUP_JOBS=$(nproc)                   # parallel jobs
export PYASC_DUMP_PATH=$PWD/dumps                  # debug dumps
```

**User context** environment variables (minimal):

```bash
export LLVM_INSTALL_PREFIX=/path/to/llvm           # required
# Everything else uses defaults (Release, no devtools, no ccache)
```

**Environment variable reference**:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLVM_INSTALL_PREFIX` | Yes | - | Path to LLVM installation |
| `PYASC_SETUP_CONFIG` | No | Release | Build config (Debug/Release/RelWithDebInfo) |
| `PYASC_SETUP_DEVTOOLS` | No | 0 | Build dev tools (ascir-opt, ascir-lsp, ascir-translate) |
| `PYASC_SETUP_CCACHE` | No | 0 | Enable ccache for faster rebuilds |
| `PYASC_SETUP_CLANG_LLD` | No | 0 | Use clang/lld instead of gcc |
| `PYASC_SETUP_DOCS` | No | 0 | Generate MLIR documentation |
| `PYASC_SETUP_JOBS` | No | auto | Number of parallel build jobs |
| `PYASC_SETUP_COVERAGE` | No | 0 | Enable code coverage |
| `PYASC_SETUP_ASAN` | No | 0 | Enable AddressSanitizer |
| `PYASC_SETUP_CMAKE_APPEND` | No | - | Additional CMake arguments |
| `PYASC_SETUP_BUILD_DIR` | No | build | Temporary build directory |
| `PYASC_HOME` | No | ~/.pyasc | Cache directory for dependencies |
| `PYASC_DUMP_PATH` | No | - | Directory for debug dumps |

---

### Phase 3: Check Installed Packages

**Goal**: Verify system and Python dependencies are installed.

#### 3.1 System Dependencies

**Check installed tools**:

```bash
gcc --version
cmake --version
ninja --version
ccache --version  # optional
clang --version   # optional
lld --version     # optional
```

> **Recommendation**: It is recommended to use `clang` and `lld` instead of `gcc` to significantly speed up the build process. Set `PYASC_SETUP_CLANG_LLD=1` to enable them.

**If any required tool is missing**:

Report which tools are missing and ask the user: "The following build tools are missing: {list}. Would you like me to install them? (requires sudo)"

Recommended installation command (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install build-essential ccache clang lld
```

#### 3.2 Python Version

Must be 3.9+:

```bash
python3 --version
```

#### 3.3 Virtual Environment

**CRITICAL**: NEVER overwrite an existing virtual environment. Always check for existing venvs first.

**Search for existing venvs in the project root**:

**CRITICAL**: `.venv` (hidden directory with dot prefix) is the MOST COMMON venv name in this project.
You MUST check for hidden directories (starting with `.`) FIRST. Do NOT rely only on glob patterns
like `*venv*` — they do NOT match hidden directories by default in bash.

```bash
# Search for ALL venvs (hidden and non-hidden)
find . -maxdepth 1 -type d \( -name '*venv*' -o -name '*pythonenv*' \) 2>/dev/null
```

If the output is empty, no venv was found.

**Decision logic based on search results**:

- **One venv found** (e.g., `.venv`): Use it automatically without asking.
  ```bash
  source .venv/bin/activate
  ```
  Then skip to Section 3.4.

- **Multiple venvs found** (e.g., `.venv`, `venv`, `env`): Ask the user which one to use.

  Ask the user: "Multiple virtual environments found in the project root: {list}. Which one should I activate?"

  After the user selects, activate it:
  ```bash
  source <selected_venv>/bin/activate
  ```
  Then skip to Section 3.4.

- **No venv found**: Ask the user whether to create one.

  Ask the user: "No virtual environment found in the project root. Would you like me to create one?"

  **BEFORE creating, verify the directory does not exist**:
  ```bash
  test -d .venv && echo "ERROR: .venv already exists" || python3 -m venv .venv --prompt pyasc
  source .venv/bin/activate
  ```

#### 3.4 Python Build Dependencies

**Check if installed**:

Read `build-system.requires` from `pyproject.toml` to get the list of build dependencies, then check with `pip list` which ones are missing.

**If missing**:

Ask the user: "Python build dependencies are missing: {list}. Would you like me to install them?"

Installation command (compose from pyproject.toml):
```bash
pip install <package1> <package2> ...
```

#### 3.5 CANN Toolkit Check (CONDITIONAL)

> **Only needed if the user requires simulator or NPU testing. Skip this step for build-only workflows.**

**Check installation**:

```bash
ls /usr/local/Ascend/cann/set_env.sh || ls ~/Ascend/cann/set_env.sh
```

**Activate**:

```bash
source /usr/local/Ascend/cann/set_env.sh
```

**For simulator**:

```bash
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH
```

**If CANN not found**:

Report the error and ask the user: "CANN toolkit is not installed. It is required for simulator/NPU testing. Would you like me to help you install it?"

See @docs/installation/setup-runtime-env.rst for detailed installation instructions.

**Failure handling**:
- System package missing: Report which packages are missing and ask the user whether to install them (requires `sudo`)
- Python version < 3.9: Report error "Python 3.9+ is required. Current version: {version}" and ask user to upgrade
- No venv: Ask user whether to create one (see Section 3.3)
- pip install fails: Report error and recommend checking network connection and pip version
- CANN not found: Report error and ask user whether to install (only if needed for testing)

---

### Phase 4: Build Project

**Goal**: Build and install PyAsc.

**CRITICAL**: Do NOT proceed with build until you have explicitly asked the user which configuration to use and received a clear answer.

Steps depend on context from Phase 2.

#### 4.1 Developer Build (recommended for contributors)

```bash
# Install build deps first
pip install -r requirements-build.txt

# Build with all developer options
export PYASC_SETUP_DEVTOOLS=1
export PYASC_SETUP_CCACHE=1
export PYASC_SETUP_CLANG_LLD=1
export PYASC_SETUP_CONFIG=Debug

# Install in editable mode with verbose output
pip install --no-build-isolation -e . -v
```

#### 4.2 Standard Build (for users)

```bash
pip install .
```

#### 4.3 Editable Install (developer, without devtools on PATH)

```bash
pip install -e .
```

#### 4.4 Build Wheel

```bash
pip wheel . --no-deps -w dist/
```

#### Verification After Build

```bash
# Check package installed
pip list | grep pyasc

# Check Python imports
python -c "import asc, asctile; print('PyAsc imported successfully')"

# Check dev tools (if PYASC_SETUP_DEVTOOLS=1)
which ascir-opt
which ascir-lsp
which ascir-translate
```

**Scope**: This skill is responsible only for environment setup and building PyAsc. Testing, debugging, and error fixing are outside the scope of this skill.

**Failure handling**:
- Build fails: Check LLVM, compiler version, try Debug config
- Import fails: Check `pip list`, verify venv active
- Tools not found: Ensure `PYASC_SETUP_DEVTOOLS=1` was set during build

---

## Agent Constraints

**CRITICAL**: The agent MUST follow these constraints:

- **NEVER** assume build configuration based on context — ALWAYS ask the user explicitly
- **DO NOT** modify source code files
- **DO NOT** install any software (apt packages, pip packages, download archives) without explicitly asking the user first
- **DO NOT** install packages in system Python (always use virtual environment)
- **DO NOT** change system settings without explicit user request
- **ALWAYS** report missing dependencies with clear error messages
- **ALWAYS** ask the user before installing any dependency or downloading any file
- **ALWAYS** check file existence before executing commands
- **ALWAYS** offer alternatives if something is not found
- **ALWAYS** verify installations after each phase
- **NEVER** run `sudo` commands without user confirmation
- **NEVER** delete or overwrite existing configurations without asking
- **NEVER** create a virtual environment if one already exists in the project root
- **ALWAYS** verify venv directory does not exist before running `python3 -m venv`
- **NEVER** report "Virtual environment not found" without first explicitly checking for hidden directories (`.venv`, `pythonenv`) using `find . -maxdepth 1 -type d \( -name '*venv*' -o -name '*pythonenv*' \)`
- **ALWAYS** use `find . -maxdepth 1 -type d \( -name '*venv*' -o -name '*pythonenv*' \) 2>/dev/null` as a standalone command — empty output means no venv found, non-empty output shows found venvs

---

## Troubleshooting

### LLVM not found

**Symptom**: `LLVM_INSTALL_PREFIX is not set` or `MLIRConfig.cmake not found`

**Solution**:
1. Download LLVM from the links in Phase 1
2. Extract to preferred location
3. Set: `export LLVM_INSTALL_PREFIX=/path/to/llvm`

### CMake not found

**Symptom**: `cmake: command not found`

**Solution**:

Report the error to the user. CMake and Ninja are installed via pip automatically during `pip install` (or from `requirements-build.txt` when `--no-build-isolation` is used).

If still missing, install build dependencies first:
```bash
pip install -r requirements-build.txt
```

### Build fails with compilation errors

**Symptom**: C++ compilation errors during `pip install`

**Solution**:
1. Check system clang version: `clang --version`
2. Ensure `PYASC_SETUP_CLANG_LLD=1` is set
3. Verify LLVM installation: `$LLVM_INSTALL_PREFIX/bin/llvm-config --version`

### CANN toolkit not found

**Symptom**: `ASCEND_HOME_PATH is not set`

**Solution**:

Report the error to the user and ask: "CANN toolkit is not installed. Would you like me to help you install it?"

Recommended steps:
1. Download CANN 9.x or newer from the official Ascend repository
2. Install: `bash Ascend-cann-toolkit_*_linux-$(arch).run --full`
3. Activate: `source /usr/local/Ascend/cann/set_env.sh`

### Simulator not working

**Symptom**: Errors when running tests with `Backend.Model`

**Solution**:
1. Check `LD_LIBRARY_PATH`: `echo $LD_LIBRARY_PATH`
2. Verify libraries: `ls $ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib/`
3. Add to environment: `export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH`

### Virtual environment issues

**Symptom**: `source .venv/bin/activate` fails

**Solution**:

Ask the user to either delete the broken virtual environment manually or provide the correct path to a working one.

### Permission denied

**Symptom**: `Permission denied` when accessing LLVM or CANN

**Solution**:

> **CRITICAL**: Ask the user before running chmod.

```bash
chmod +x $LLVM_INSTALL_PREFIX/bin/*
chmod +x /usr/local/Ascend/cann/set_env.sh
```
