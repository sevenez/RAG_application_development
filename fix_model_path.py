#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

# 要修改的文件路径
file_path = "04-Question-Answering-System/09-hybrid-retrieval.py"

# 读取文件内容
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换模型路径设置
old_code = '''        # 修改本地模型存储路径 - 使用原始字符串避免转义问题
        self.local_model_dir = r"D:\\\\rag-project\\\\05-rag-practice\\\\11_local_models"
        self.local_bge_model_dir = os.path.join(self.local_model_dir, "bge-m3")
        self.local_cross_encoder_dir = os.path.join(self.local_model_dir, "cross-encoder-ms-marco")'''

new_code = '''        # 使用智能模型路径检测（参考08-legal-vector-db.py）
        # 优先使用用户缓存目录中的模型
        user_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        # Hugging Face缓存目录结构：models--{org}--{model_name}/snapshots/{commit_hash}
        model_cache_dir = os.path.join(user_cache_dir, "models--BAAI--bge-m3")
        
        # 查找最新的快照目录
        if os.path.exists(model_cache_dir):
            snapshots_dir = os.path.join(model_cache_dir, "snapshots")
            if os.path.exists(snapshots_dir):
                # 获取最新的快照目录
                snapshot_dirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshot_dirs:
                    # 使用最新的快照（按修改时间排序）
                    latest_snapshot = max(snapshot_dirs, key=lambda x: os.path.getmtime(os.path.join(snapshots_dir, x)))
                    self.local_bge_model_dir = os.path.join(snapshots_dir, latest_snapshot)
                else:
                    self.local_bge_model_dir = model_cache_dir
            else:
                self.local_bge_model_dir = model_cache_dir
        else:
            # 如果缓存目录不存在，使用项目本地模型目录
            self.local_bge_model_dir = os.path.join(os.path.dirname(os.getcwd()), "11_local_models", "bge-m3")
        
        # cross-encoder模型路径
        self.local_cross_encoder_dir = os.path.join(os.path.dirname(os.getcwd()), "11_local_models", "cross-encoder-ms-marco")'''

# 执行替换
content = content.replace(old_code, new_code)

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("模型路径修复完成！")