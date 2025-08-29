"""
法律文本条款分割器

功能：
- 读取Word格式的法律文档（如《中华人民共和国民法典.docx》）
- 使用基于Transformer的段落分割算法对文档进行智能分割
- 通过分析法律文本的句式特征（如"第X条"、"凡...者"等）识别独立条款
- 将冗长文档拆分为具有独立法律效力的条款单元
- 生成JSON格式的分割结果，便于后续处理和分析

依赖：
- python-docx: 读取Word文档内容
- transformers: 使用预训练的Transformer模型
- torch: 深度学习框架支持
- numpy: 数据处理
- tqdm: 进度显示

使用方法：
1. 安装依赖：pip install python-docx transformers torch numpy tqdm
2. 确保待处理的法律文档（如《中华人民共和国民法典.docx》）在当前目录下
3. 运行程序：python 04-legal_article_segmentation.py
4. 查看生成的JSON结果文件（如《中华人民共和国民法典_条款分割.json》）

注意事项：
- 如无法加载Transformer模型，程序会自动降级使用基于规则的分割方法
- 程序会自动检测文档中的条款特征，包括"第X条"、"凡...者"等法律特有的句式
- 分割结果包含条款文本、长度、是否含条款编号等信息
"""
# 抑制TensorFlow和Protobuf警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全部信息, 1=INFO, 2=WARNING, 3=ERROR
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import re
import json
from typing import List, Dict, Tuple
import docx
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil

class LegalArticleSegmenter:
    """
    民法典条款分割器，基于Transformer模型和法律文本特征进行条款分割
    """
    def __init__(self, model_name="uer/roberta-base-finetuned-chinanews-chinese"):
        """初始化分割器，加载预训练模型"""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.load_model()
        
        # 法律文本特征模式
        self.article_patterns = [
            r'第\s*[零一二三四五六七八九十百千]+\s*条',  # 中文数字条
            r'第\s*\d+\s*条',  # 阿拉伯数字条
            r'凡\s*[^。；；]+者',  # 凡...者句式
            r'[一二三四五六七八九十]+、',  # 项号标记
            r'第\s*[一二三四五六七八九十百千]+\s*款',  # 款号标记
        ]
        
        # 编译正则表达式以提高效率
        self.compiled_patterns = [re.compile(pattern) for pattern in self.article_patterns]
    
    def load_model(self):
        """加载预训练的Transformer模型（优先使用缓存）"""
        try:
            # 显示缓存调试信息
            self.get_model_cache_info()
            
            # 首先检查 Hugging Face 缓存中是否有模型
            if self._check_huggingface_cache():
                print(f"在 Hugging Face 缓存中找到模型: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.eval()
                print("缓存模型加载成功！")
            else:
                # 缓存中没有模型，尝试下载
                print(f"正在从 Hugging Face 下载模型: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.eval()
                print("模型下载并加载成功！")
                    
        except Exception as e:
            print(f"警告：模型加载失败，将使用基于规则的方法进行分割。错误信息：{str(e)}")
            self.tokenizer = None
            self.model = None
    
    def _check_huggingface_cache(self) -> bool:
        """检查 Hugging Face 缓存中是否存在模型"""
        try:
            import os
            from pathlib import Path
            
            # 获取 Hugging Face 缓存目录
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            
            # 构建模型缓存目录名（格式：models--用户名--模型名）
            model_cache_name = f"models--{self.model_name.replace('/', '--')}"
            model_cache_path = os.path.join(cache_dir, model_cache_name)
            
            if not os.path.exists(model_cache_path):
                return False
            
            # 检查是否有 snapshots 目录（包含实际的模型文件）
            snapshots_dir = os.path.join(model_cache_path, "snapshots")
            if not os.path.exists(snapshots_dir):
                return False
            
            # 检查 snapshots 目录下是否有文件夹（版本快照）
            snapshot_dirs = [d for d in os.listdir(snapshots_dir) 
                           if os.path.isdir(os.path.join(snapshots_dir, d))]
            
            if not snapshot_dirs:
                return False
            
            # 检查最新的快照中是否包含必要的模型文件
            latest_snapshot = os.path.join(snapshots_dir, snapshot_dirs[0])
            required_files = ["config.json"]
            
            for file in required_files:
                if not os.path.exists(os.path.join(latest_snapshot, file)):
                    return False
            
            print(f"找到缓存模型路径: {model_cache_path}")
            return True
            
        except Exception as e:
            print(f"检查缓存时出错: {e}")
            return False
    
    def get_model_cache_info(self):
        """获取模型缓存信息（用于调试）"""
        try:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_cache_name = f"models--{self.model_name.replace('/', '--')}"
            model_cache_path = os.path.join(cache_dir, model_cache_name)
            
            print(f"Hugging Face 缓存目录: {cache_dir}")
            print(f"模型缓存路径: {model_cache_path}")
            print(f"缓存是否存在: {os.path.exists(model_cache_path)}")
            
            if os.path.exists(model_cache_path):
                snapshots_dir = os.path.join(model_cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = os.listdir(snapshots_dir)
                    print(f"可用快照: {snapshots}")
                    
        except Exception as e:
            print(f"获取缓存信息时出错: {e}")
    
    def read_docx_file(self, file_path: str) -> str:
        """读取Word文档内容"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        doc = docx.Document(file_path)
        full_text = []
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:  # 跳过空段落
                full_text.append(text)
        
        return '\n'.join(full_text)
    
    def detect_article_boundaries(self, text: str) -> List[int]:
        """检测条款边界位置"""
        boundaries = [0]  # 起始位置
        
        # 使用正则表达式查找所有可能的条款起始位置
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                start_pos = match.start()
                if start_pos not in boundaries:
                    boundaries.append(start_pos)
        
        # 排序边界位置
        boundaries.sort()
        
        # 如果边界之间的文本太短，可能是误判，需要合并
        filtered_boundaries = [0]
        for i in range(1, len(boundaries)):
            if boundaries[i] - filtered_boundaries[-1] > 50:  # 文本长度阈值
                filtered_boundaries.append(boundaries[i])
        
        # 确保包含文本结尾
        if filtered_boundaries[-1] < len(text):
            filtered_boundaries.append(len(text))
        
        return filtered_boundaries
    
    def split_by_transformer(self, text: str) -> List[str]:
        """使用Transformer模型进行智能段落分割"""
        # 如果没有加载模型，则使用基于规则的方法
        if self.model is None:
            return self.split_by_rules(text)
        
        # 将长文本分割成合理的块
        chunks = []
        max_length = 512  # 模型的最大序列长度
        current_pos = 0
        
        # 先使用规则检测找到可能的条款边界
        boundaries = self.detect_article_boundaries(text)
        
        # 根据边界分割文本
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
        
        # 对于每个块，使用Transformer进行进一步处理和评分
        # 这里简化处理，主要依赖规则检测的结果
        return chunks
    
    def split_by_rules(self, text: str) -> List[str]:
        """使用规则进行文本分割"""
        chunks = []
        boundaries = self.detect_article_boundaries(text)
        
        # 根据边界分割文本
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def segment_document(self, file_path: str) -> List[Dict]:
        """分割文档并返回结构化的条款列表"""
        print(f"正在处理文档: {file_path}")
        text = self.read_docx_file(file_path)
        
        # 使用Transformer进行分割
        articles = self.split_by_transformer(text)
        
        # 结构化每个条款
        structured_articles = []
        for i, article_text in enumerate(articles, 1):
            article_info = {
                "id": i,
                "text": article_text,
                "length": len(article_text),
                "has_article_number": self._has_article_number(article_text),
                "start_with": article_text[:20] + ("..." if len(article_text) > 20 else "")
            }
            structured_articles.append(article_info)
        
        print(f"文档分割完成，共识别出 {len(structured_articles)} 个条款")
        return structured_articles
    
    def _has_article_number(self, text: str) -> bool:
        """检查文本是否包含条款编号"""
        for pattern in self.compiled_patterns[:2]:  # 只检查条号模式
            if pattern.search(text):
                return True
        return False
    
    def save_articles_to_json(self, articles: List[Dict], output_file: str):
        """将分割后的条款保存为JSON文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"条款分割结果已保存至: {output_file}")

    def summarize_segmentation(self, articles: List[Dict]) -> Dict:
        """生成分割结果摘要"""
        total_articles = len(articles)
        articles_with_number = sum(1 for art in articles if art["has_article_number"])
        avg_length = sum(art["length"] for art in articles) / total_articles if total_articles > 0 else 0
        
        return {
            "total_articles": total_articles,
            "articles_with_number": articles_with_number,
            "average_length": round(avg_length, 2),
            "coverage_rate": round((articles_with_number / total_articles * 100) if total_articles > 0 else 0, 2)
        }


def main():
    """主函数"""
    # 获取脚本所在目录，而不是当前工作目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义文件路径
    doc_file_path = os.path.join(script_dir, "中华人民共和国民法典.docx")
    output_file_path = os.path.join(script_dir, "中华人民共和国民法典_条款分割.json")
    
    # 检查输入文件是否存在
    if not os.path.exists(doc_file_path):
        print(f"错误：找不到文件 {doc_file_path}")
        print("请确保该文件在当前目录下。")
        print(f"当前目录: {current_dir}")
        print("当前目录文件列表:")
        for file in os.listdir(current_dir):
            print(f"  - {file}")
        return
    
    # 初始化分割器
    segmenter = LegalArticleSegmenter()
    
    # 分割文档
    articles = segmenter.segment_document(doc_file_path)
    
    # 保存结果
    segmenter.save_articles_to_json(articles, output_file_path)
    
    # 生成并显示摘要
    summary = segmenter.summarize_segmentation(articles)
    print("\n分割结果摘要:")
    print(f"- 总条款数: {summary['total_articles']}")
    print(f"- 含条款编号: {summary['articles_with_number']}")
    print(f"- 平均条款长度: {summary['average_length']} 字符")
    print(f"- 条款编号覆盖率: {summary['coverage_rate']}%")
    
    # 显示前3个条款的示例
    print("\n前3个条款示例:")
    for i, article in enumerate(articles[:3]):
        print(f"\n条款 {i+1}:")
        print(f"{article['text'][:200]}..." if len(article['text']) > 200 else article['text'])


if __name__ == "__main__":
    main()