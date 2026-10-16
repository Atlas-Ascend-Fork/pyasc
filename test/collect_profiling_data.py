# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import csv
import os

columns = ['File Name', 'Task Duration(us)']
with open('./all_task_durations.csv', 'w', newline='', encoding='utf-8') as output:
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    csv_files = [f for f in os.listdir('./copy_csv/') if f.endswith('.csv')]
    for file_name in csv_files:
        file_path = os.path.join('./copy_csv/', file_name)
        with open(file_path, newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            if 'Task Duration(us)' not in (reader.fieldnames or []):
                continue
            writer.writerows({'File Name': file_name, 'Task Duration(us)': row['Task Duration(us)']} for row in reader)

print("Save result to all_task_durations.csv.")
