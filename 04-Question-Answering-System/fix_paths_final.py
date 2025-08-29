#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# 读取原始文件
file_path = "04-Question-Answering-System/08-legal-vector-db.py"
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到要修改的行
new_lines = []
for i, line in enumerate(lines):
    if 'directory_path = r"d:\\\\rag-project\\\\05-rag-practice\\\\04-Question-Answering-System"' in line:
        # 替换为相对路径
        new_lines.append('        # 使用相对路径访问法规目录\n')
        new_lines.append('        法规_dir = os.path.join("..", "20-Data", "03-法规")\n')
        new_lines.append('        if os.path.exists(法规_dir):\n')
        new_lines.append('            target_directory = 法规_dir\n')
        new_lines.append('            print(f"使用法规目录: {法规_dir}")\n')
        new_lines.append('        else:\n')
        new_lines.append('            print(f"无法访问法规目录 {法规_dir}，将使用当前目录")\n')
        new_lines.append('            target_directory = "."\n')
        
        # 跳过原来的几行
        skip_count = 7  # 跳过原来的7行代码
        for j in range(i+1, min(i+1+skip_count, len(lines))):
            if 'target_directory = directory_path' in lines[j]:
                break
    else:
        new_lines.append(line)

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("文件路径已成功修改为相对路径！")