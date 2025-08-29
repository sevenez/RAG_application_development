#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# 测试相对路径
print("测试相对路径访问...")

# 测试法规目录
法规_dir = os.path.join("..", "20-Data", "03-法规")
print(f"法规目录路径: {法规_dir}")
print(f"目录是否存在: {os.path.exists(法规_dir)}")

# 测试当前目录
current_dir = "."
print(f"当前目录路径: {os.path.abspath(current_dir)}")
print(f"当前目录是否存在: {os.path.exists(current_dir)}")

# 列出当前目录内容
print("\n当前目录内容:")
for item in os.listdir("."):
    if os.path.isdir(item):
        print(f"📁 {item}/")
    else:
        print(f"📄 {item}")

print("\n相对路径测试完成！")