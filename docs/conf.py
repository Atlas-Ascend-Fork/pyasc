# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import datetime
import importlib.util
import os
import re
import tempfile
from pathlib import Path

package_root = Path(os.pardir).resolve()


def import_setup_tool():
    spec = importlib.util.spec_from_file_location("setup_tool", package_root / "setup.py")
    setup_tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup_tool)
    return setup_tool


def render_toc(src_file, dst_file, heading_levels):
    preamble = []
    content = []
    read_content = False
    with open(src_file) as f:
        line = f.readline()
        while line:
            if "[TOC]" in line:
                read_content = True
                continue
            if read_content:
                content.append(line)
            else:
                preamble.append(line)
            line = f.readline()
    toc = []
    re_heading = re.compile(r"^(#+) (.+)$")
    re_anchor = re.compile(r"[^0-9a-zA-Z]+")
    for line in content:
        m = re_heading.match(line)
        if m is None:
            continue
        level = len(m.group(1))
        if level not in heading_levels:
            continue
        heading = m.group(2)
        anchor = re_anchor.sub("-", heading).lower().strip(" -")
        toc_line = "  " * (level - 2)
        toc_line += f"* [{heading}](#{anchor})\n"
        toc.append(toc_line)
    with open(dst_file, "w") as f:
        f.writelines(preamble)
        f.writelines(toc)
        f.writelines(content)


def setup_generated_mlir_docs():
    setup_tool = import_setup_tool()
    cmake_dir = setup_tool.get_cmake_dir()
    src_docs_dir = cmake_dir / "docs"
    if not os.path.isdir(src_docs_dir):
        raise RuntimeError(f"{src_docs_dir} must exist")

    dialects_docs = [
        ("Dialects/AscendC.md", "ascendc.md"),
        ("Dialects/EmitAsc.md", "emitasc.md"),
    ]
    mlir_docs_dir = Path("mlir")
    for src_md, dst_md in dialects_docs:
        render_toc(src_docs_dir / src_md, mlir_docs_dir / "dialects" / dst_md, (2, 3))

    passes_docs = [
        ("AscendCPasses.md", "ascendc.md", "AscendC"),
    ]
    for src_md, dst_md, title in passes_docs:
        # Add top-level heading, TOC, and descrease other headings level by 1
        temp_fd, temp_md = tempfile.mkstemp()
        with open(src_docs_dir / src_md) as src, open(temp_fd, "w") as dst:
            dst.write(f"# {title} passes\n[TOC]\n")
            line = src.readline()
            while line:
                if line.startswith("###"):
                    dst.write(line[1:])
                else:
                    dst.write(line)
                line = src.readline()
        render_toc(temp_md, mlir_docs_dir / "passes" / dst_md, (2, ))
        os.unlink(temp_md)


def autodoc_process_signature(app, what, name: str, obj, options, signature: str, return_annotation):
    if name.startswith("asc.language.") and signature and "builder" in signature:
        signature = signature.split("builder")[0] + ")"
    return signature, return_annotation


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PYASC'
current_year = datetime.datetime.now().year
copyright = f'2025-{current_year}, Huawei Technologies Co., Ltd.'
author = 'Huawei Technologies Co., Ltd.'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]
autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'development', 'installation', 'mlir']
napoleon_preprocess_types = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]

suppress_warnings = [
    "myst.xref_missing",
]


def setup(app):
    app.connect("autodoc-process-signature", autodoc_process_signature)
