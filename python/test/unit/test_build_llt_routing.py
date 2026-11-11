# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestBuildLltRouting(unittest.TestCase):
    ALL = ("true", "all", "true", "all")
    SKIP = ("false", "all", "false", "all")
    CPP = ("true", "Adv", "false", "all")
    PYTHON = ("false", "all", "true", "language/core")
    CPP_FILE = "lib/Target/AscendC/Adv/Matmul.cpp"
    PYTHON_FILE = "python/asc/language/core/types.py"

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="pyasc_routing_")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        source_path = Path(__file__).resolve().parents[3] / "test" / "build_llt.sh"
        source = source_path.read_text(encoding="utf-8")
        # Load the production declarations and functions before the CLI starts.
        functions, separator, _ = source.partition('\nTEST="all"\n')
        if not separator:
            raise AssertionError("build_llt.sh CLI entry was not found")
        cls.functions = cls.directory / "functions.sh"
        cls.functions.write_text(functions, encoding="utf-8")
        # Initialization only queries pybind11's CMake path; routing needs no SDK.
        (cls.directory / "pybind11.py").write_text(
            "def get_cmake_dir():\n    return " + repr(str(cls.directory)) + "\n", encoding="utf-8")
        cls.environment = dict(os.environ, PYTHONPATH=str(cls.directory))

    def route(self, files):
        filelist = self.directory / "files.txt"
        if filelist.exists():
            filelist.unlink()
        if files is not None:
            filelist.write_text("\n".join(files), encoding="utf-8")
        command = '''source "$1"
PR_FILELIST="$2"
analyze_pr_filelist
printf 'ROUTING=%s|%s|%s|%s\\n' "$NEED_CPP_TEST" "$CPP_TEST_TARGET" "$NEED_PYTHON_TEST" "$PYTHON_TEST_TARGET"
'''
        result = subprocess.run(
            ["bash", "-c", command, "routing", str(self.functions),
             str(filelist)], cwd=self.directory, env=self.environment, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ROUTING=", result.stdout)
        return tuple(result.stdout.rsplit("ROUTING=", 1)[1].strip().split("|"))

    def test_cmake_files_are_excluded_from_whitelist(self):
        for filename in ("CMakeLists.txt", "cmake/CMakeLists.txt", "lib/Target/AscendC/Adv/CMakeLists.txt"):
            with self.subTest(filename=filename):
                result = subprocess.run([
                    "bash", "-c", 'source "$1"; if is_in_whitelist "$2"; then exit 1; fi', "whitelist",
                    str(self.functions), filename
                ], cwd=self.directory, env=self.environment, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_full_test_paths(self):
        for filename in ("CMakeLists.txt", "cmake/CMakeLists.txt", "cmake/options.cmake", "include/kernel.h"):
            with self.subTest(filename=filename):
                self.assertEqual(self.route([filename]), self.ALL)

    def test_full_path_overrides_module_selection_in_either_order(self):
        for module_file in (self.CPP_FILE, self.PYTHON_FILE):
            for files in (["CMakeLists.txt", module_file], [module_file, "CMakeLists.txt"]):
                with self.subTest(files=files):
                    self.assertEqual(self.route(files), self.ALL)

    def test_documentation_and_plain_text_skip_tests(self):
        for files in (["docs/README.md"], ["NOTES.txt"], ["docs/guide.rst",
                                                          "NOTES.txt"], ["cmake/options.txt", "include/platform.txt"]):
            with self.subTest(files=files):
                self.assertEqual(self.route(files), self.SKIP)

    def test_single_module_selects_precise_tests(self):
        for filename, expected in ((self.CPP_FILE, self.CPP), (self.PYTHON_FILE, self.PYTHON),
                                   ("lib/Target/AscendC/Adv/CMakeLists.txt", self.CPP)):
            with self.subTest(filename=filename):
                self.assertEqual(self.route([filename]), expected)

    def test_missing_filelist_runs_all_tests(self):
        self.assertEqual(self.route(None), self.ALL)

    def test_empty_filelist_skips_tests(self):
        self.assertEqual(self.route([]), self.SKIP)


if __name__ == "__main__":
    unittest.main()
