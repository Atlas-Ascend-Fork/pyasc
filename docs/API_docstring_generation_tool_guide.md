# API文档自动生成工具使用指南

## 概述

本项目采用[Sphinx](https://sphinxsearch.com/)工具，通过提取项目 Python 代码（如模块、类、函数）中的 `docstring`（文档字符串）自动生成标准化的 Python API 文档。该文档可清晰展示代码的功能说明、参数定义、返回值类型及使用示例，旨在降低文档维护成本，确保代码与文档的一致性。当前已在项目根目录下创建 `docs` 目录，并完成核心配置文件（`conf.py`、`index.rst`、`Makefile` 等）的初始化，用户可直接基于现有框架完成文档的更新与生成。

## 环境准备

以开发者模式或普通模式安装pyasc包、文档生成工具所需的依赖。

- 开发者模式

    在项目根目录下执行如下命令，将同时安装pyasc包和文档生成工具所需的依赖。

    ```shell
    pip install -e '.[docs]'
    ```

- 普通模式

    在项目根目录下执行如下命令，分别安装pyasc包、Sphinx及相关插件。

    ```shell
    pip install .
    pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
    pip install sphinx-markdown-builder
    ```

## 文档生成核心文件说明

`docs` 目录下的关键文件功能如下，请勿随意删除核心文件，如需修改请参考下方指引。

| 文件名           | 作用说明                                           |
| ------------- | ---------------------------------------------- |
| `conf.py`     | Sphinx 核心配置文件，包含文档主题、扩展插件、项目信息（如名称、版本）等配置      |
| `index.rst`   | 文档首页入口文件，定义文档的目录结构（如模块列表、子文档链接）                |
| `Makefile`    | 自动化脚本文件，提供 `make markdown` 和 `make html` 等命令用于快速生成文档               |
| `python-api`  | Python前段模块索引文件，用于聚合项目中所有待提取 docstring 的 Python 模块                |
| `_static`     | 配置相关css格式，仅影响html静态网页生成格式                                               |

## 标准 API 文档生成流程

首次生成或全量更新文档时，执行以下步骤：

1. **进入 docs 目录**：在项目根目录下打开终端，切换到 `docs` 目录。

    ```shell
    cd docs
    ```

2. **（可选）更新模块索引**：若项目新增或删除了 Python 模块，需更新 `python-api`目录下对应的rst格式模块索引文件，确保该模块已被包含或已删除。 `rst` 文件格式示例如下。

    ```shell
    项目模块列表
    ===========

    .. toctree::
    :maxdepth: 4

    模块1路径（如 src.utils）
    模块2路径（如 src.core）
    ```

3. **生成文档**：执行 `Makefile` 中的 `markdown` 或 `html` 命令，Sphinx 会自动提取 `docstring` 并生成 Markdown 文档或 html 静态网页。根据需要执行相应命令。

    ```shell
    # Linux/Mac 环境
    make markdown   # 生成markdown格式文档
    make html       # 生成html格式文档
    # Windows 环境（无 Make 工具时）
    sphinx-build -b markdown . _build/markdown     # 生成markdown格式文档
    sphinx-build -b html . _build/html             # 生成html格式文档
    ```

4.  **查看生成的文档**：文档生成成功后，会保存在 `docs/_build/markdown` 或 `docs/_build/html` 目录下，直接打开 `index.md` 或 `index.html` 文件，即可查看完整的 API 文档。

## 更新API文档流程

当用户修改了 Python 文件（如新增函数、更新 `docstring` 内容），无需执行API文档生成的全量流程，仅需执行以下简化步骤：

1.  **确认docstring内容**：确保修改后的函数 / 类 / 模块已按规范编写 `docstring`（推荐 Google 风格或 NumPy 风格，规范细节可参考[Google 风格官方指南](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)、[NumPy 风格官方指南](https://numpydoc.readthedocs.io/en/latest/format.html)），docstring内容示例如下。

    ```python
    def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
        """
        按元素求和。

        **对应的Ascend C函数原型**

        .. code-block:: c++

            template <typename T>
            __aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                        const LocalTensor<T>& src1, const int32_t& count);

        .. code-block:: c++

            template <typename T, bool isSetMask = true>
            __aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                        const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTimes,
                                        const BinaryRepeatParams& repeatParams);
                                        
        .. code-block:: c++

            template <typename T, bool isSetMask = true>
            __aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                        const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTimes,
                                        const BinaryRepeatParams& repeatParams);

        
        **参数说明**

        - is_set_mask：是否在接口内部设置mask模式和mask值。
        - dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
        - src0, src1: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
        - count: 参与计算的元素个数。
        - mask: 用于控制每次迭代内参与计算的元素。
        - repeat_times: 重复迭代次数。
        - params: 控制操作数地址步长的参数。

        **返回值说明**（若无则无需补充）
        ...

        **调用示例**

        - tensor高维切分计算样例-mask连续模式

          .. code-block:: python

              mask = 128
              # repeat_times = 4，一次迭代计算128个数，共计算512个数
              # dst_blk_stride, src0_blk_stride, src1_blk_stride = 1，单次迭代内数据连续读取和写入
              # dst_rep_stride, src0_rep_stride, src1_rep_stride = 8，相邻迭代间数据连续读取和写入
              params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
              asc.add(dst, src0, src1, mask=mask, repeat_times=4, repeat_params=params)

        - tensor高维切分计算样例-mask逐bit模式

          .. code-block:: python

              mask = [uint64_max, uint64_max]
              # repeat_times = 4，一次迭代计算128个数，共计算512个数
              # dst_blk_stride, src0_blk_stride, src1_blk_stride = 1，单次迭代内数据连续读取和写入
              # dst_rep_stride, src0_rep_stride, src1_rep_stride = 8，相邻迭代间数据连续读取和写入
              params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
              asc.add(dst, src0, src1, mask=mask, repeat_times=4, repeat_params=params)

        - tensor前n个数据计算样例

          .. code-block:: python

              asc.add(dst, src0, src1, count=512)
    
        """
        builder = global_builder.get_ir_builder()
        op_impl("add", dst, src0, src1, args, kwargs, builder.create_asc_AddL0Op, builder.create_asc_AddL1Op,
                builder.create_asc_AddL2Op)
    ```

2.  **进入docs目录并重新生成文档**：在 `docs` 目录下执行如下文档生成命令，Sphinx 将自动识别代码变更并更新文档内容。根据需要执行相应命令。

    ```shell
    # Linux/Mac 环境
    cd docs
    make markdown   # 更新markdown格式文档
    make html       # 更新html格式文档
    # Windows 环境
    cd docs 
    sphinx-build -b markdown . _build/markdown     # 更新markdown格式文档
    sphinx-build -b html . _build/html             # 更新html格式文档
    ```

3.  **验证文档更新结果**：重新打开 `docs/_build/markdown/index.md` 或 `docs/_build/html/index.html`，导航到该修改的模块 / 函数，确认文档内容已同步更新。

## 常见问题（FAQ）

1.  **问题**：执行 `make markdown` 时提示 “No module named xxx”

    **解决**：确保待提取的 Python 模块已在项目环境变量中，可在 `conf.py` 中添加路径配置。

    ```shell
    import os
    import sys
    # 将项目根目录添加到 Python 路径（根据实际结构调整）
    sys.path.insert(0, os.path.abspath('../src'))
    ```

2.  **问题**：生成的文档中缺少部分函数 / 类的说明

    **解决**：检查以下两点：

    *   确认函数 / 类的 `docstring` 格式符合 Sphinx 识别规范（参考上述Google 风格或 NumPy 风格规范）；
    *   确认对应的模块已添加到`python-api` 目录中对应rst文件的 `toctree` 列表中。

3. **问题**：生成的文档中内容格式显示错误

    **解决**：检查docstring中内容对齐格式。

    ```python
    """
    **参数说明**

    - mode_number：合入位置参数，取值范围：modeNumber∈[0, 5]
      # 不同层级之间添加一个空行
      - 0：合入x1       # 当前层级标题应与上一层级标题正文内容对齐
      - 1：合入y1
      - 2：合入x2
      - 3：合入y2
      - 4：合入score
      - 5：合入label

    **调用示例**
    
    .. code-block:: python  # 代码块应与上一行正文内容对齐

        a = 1
        b = 2
        c = a + b

    - tensor前n个数据计算样例

      .. code-block:: python    # 代码块应与上一行正文内容对齐

          asc.add(dst, src0, src1, count=512)
    """
    ```

## 扩展与定制

若需自定义文档风格或功能，可修改以下配置：

*   **支持更多 docstring 风格**：在 `conf.py` 的 `extensions` 中添加 `sphinx.ext.napoleon`（支持 Google/NumPy 风格 docstring）。
