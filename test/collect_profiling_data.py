# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import csv
import os
import stat
import tempfile

columns = ['File Name', 'Task Duration(us)']
csv_files = [f for f in os.listdir('./copy_csv/') if f.endswith('.csv')]
output_path = os.path.realpath('./all_task_durations.csv')
try:
    output_fd = os.open(output_path, os.O_WRONLY)
except FileNotFoundError:
    output_mode = None
else:
    try:
        output_mode = stat.S_IMODE(os.fstat(output_fd).st_mode)
    finally:
        os.close(output_fd)

with tempfile.TemporaryDirectory(prefix='.profiling-summary-', dir=os.path.dirname(output_path)) as temporary_dir:
    temporary_path = os.path.join(temporary_dir, 'all_task_durations.csv')
    with open(temporary_path, 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for file_name in csv_files:
            file_path = os.path.join('./copy_csv/', file_name)
            with open(file_path, newline='', encoding='utf-8-sig') as stream:
                reader = csv.DictReader(stream)
                if 'Task Duration(us)' not in (reader.fieldnames or []):
                    continue
                writer.writerows({'File Name': file_name, 'Task Duration(us)': row['Task Duration(us)']}
                                 for row in reader)
    if output_mode is not None:
        os.chmod(temporary_path, output_mode)
    os.replace(temporary_path, output_path)

print("Save result to all_task_durations.csv.")
