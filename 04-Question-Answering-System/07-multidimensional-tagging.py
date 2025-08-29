#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""法律条款多维度标签体系自动标注系统

功能概述：
该系统用于构建多维度标签体系，通过半监督学习方法对法律条款进行自动分类标注，
支持"法规类型"、"效力等级"、"管辖区域"等多个维度的标签体系，可处理百万级条款数据。

主要功能：
- 定义多维度标签树结构
- 实现基于规则的初始标签标注
- 构建半监督学习模型进行自动分类
- 支持百万级条款的批量处理
- 与现有哈希值和元数据系统集成
- 提供标注结果可视化和评估功能

使用方法：
1. 准备已分割的法律条款JSON文件
2. 配置标签体系和模型参数
3. 运行程序进行自动标注
4. 查看和导出标注结果

输入输出：
输入：已分割的法律条款JSON文件（如'中华人民共和国民法典_条款分割.json'）
输出：
  - 带多维度标签的法律条款JSON（如'中华人民共和国民法典_标签标注.json'）
  - 标签统计和评估报告
"""

import os
# 隐藏 TensorFlow 日志信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 隐藏 INFO 和 WARNING 日志
import warnings
# 屏蔽 matplotlib 字体警告
warnings.filterwarnings('ignore', message='.*Font family.*not found.*')
warnings.filterwarnings('ignore', module='matplotlib.font_manager')
warnings.filterwarnings('ignore', category=UserWarning, message='.*findfont.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*Font family.*')

# 进一步屏蔽 Matplotlib 的 findfont 日志
import logging
import matplotlib as mpl
mpl.set_loglevel('error')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import re
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import hashlib
import time
from tqdm import tqdm
import concurrent.futures
from typing import List, Dict, Tuple, Set, Any

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

class LegalMultiDimensionalTagger:
    """法律条款多维度标签自动标注系统"""
    
    def __init__(self):
        """初始化标签系统，定义多维度标签树"""
        # 获取脚本所在目录
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 多维度标签体系定义
        self.tag_trees = {
            "法规类型": {
                "基本法": ["宪法", "民法典", "刑法典"],
                "单行法": ["劳动法", "合同法", "物权法", "侵权责任法", "知识产权法"],
                "行政法规": ["条例", "规定", "办法", "细则"],
                "地方法规": [],
                "司法解释": [],
                "部门规章": [],
                "国际条约": []
            },
            "效力等级": {
                "宪法": [],
                "法律": [],
                "行政法规": [],
                "地方法规": [],
                "部门规章": [],
                "规范性文件": []
            },
            "管辖区域": {
                "全国性": ["中央", "全国"],
                "地方": {
                    "华北": ["北京", "天津", "河北", "山西", "内蒙古"],
                    "东北": ["辽宁", "吉林", "黑龙江"],
                    "华东": ["上海", "江苏", "浙江", "安徽", "福建", "江西", "山东"],
                    "华南": ["广东", "广西", "海南"],
                    "华中": ["河南", "湖北", "湖南"],
                    "西南": ["重庆", "四川", "贵州", "云南", "西藏"],
                    "西北": ["陕西", "甘肃", "青海", "宁夏", "新疆"],
                    "港澳台": ["香港", "澳门", "台湾"]
                }
            },
            "内容领域": {
                "民事": ["合同", "物权", "侵权", "婚姻家庭", "继承"],
                "刑事": ["犯罪", "刑罚", "刑事诉讼"],
                "行政": ["行政许可", "行政处罚", "行政强制", "行政复议"],
                "经济": ["公司", "金融", "税收", "知识产权"],
                "劳动": ["劳动合同", "工资福利", "劳动争议"],
                "环境": [],
                "诉讼": ["民事诉讼", "刑事诉讼", "行政诉讼"]
            }
        }
        
        # 标签规则库
        self.tag_rules = {
            "法规类型": {
                "宪法": [r'宪法'],
                "民法典": [r'民法典'],
                "刑法典": [r'刑法典'],
                "劳动法": [r'劳动法'],
                "合同法": [r'合同法'],
                "物权法": [r'物权法'],
                "条例": [r'条例'],
                "规定": [r'规定'],
                "办法": [r'办法'],
                "细则": [r'细则']
            },
            "效力等级": {
                "宪法": [r'宪法'],
                "法律": [r'法$'],
                "行政法规": [r'条例'],
                "地方法规": [r'([省市自治区]|[京津沪渝])[^，,]+条例'],
                "部门规章": [r'规定|办法|细则'],
                "规范性文件": [r'通知|意见|决定']
            },
            "管辖区域": {
                "全国性": [r'中华人民共和国', r'全国', r'中央'],
                "北京": [r'北京市'],
                "上海": [r'上海市'],
                "广东": [r'广东省'],
                "江苏": [r'江苏省'],
                # 其他地区规则...
            },
            "内容领域": {
                "合同": [r'合同|协议|契约'],
                "物权": [r'所有权|用益物权|担保物权|不动产|动产'],
                "侵权": [r'侵权|损害赔偿|过错责任'],
                "婚姻家庭": [r'结婚|离婚|夫妻|子女|家庭'],
                "继承": [r'继承|遗嘱|遗产'],
                "公司": [r'公司|企业|股东|董事'],
                "劳动合同": [r'劳动合同|雇佣|劳动者|用人单位'],
                "知识产权": [r'商标|专利|著作权|知识产权']
            }
        }
        
        # 编译规则的正则表达式以提高性能
        self.compiled_rules = {}
        for dimension, tags in self.tag_rules.items():
            self.compiled_rules[dimension] = {}
            for tag_name, patterns in tags.items():
                self.compiled_rules[dimension][tag_name] = [re.compile(pattern) for pattern in patterns]
        
        # 模型参数
        self.max_seq_length = 512
        self.embedding_dim = 256
        self.lstm_units = 128
        self.batch_size = 32
        self.epochs = 10
        
        # 数据存储
        self.articles = []
        self.tagged_articles = []
        self.metadata_store = {}
        
        # 初始化模型
        self.models = {}
        self.tokenizer = Tokenizer(num_words=50000, oov_token="<UNK>")
        
        # 并行处理参数
        self.max_workers = min(10, os.cpu_count() or 4)
        
    def _generate_sha256(self, text: str) -> str:
        """为文本生成SHA-256哈希值"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def load_articles(self, file_path: str, source_institution: str = "未知机构", 
                      revision_history: List[Dict] = None) -> List[Dict]:
        """加载已分割的法律条款文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        print(f"正在加载条款文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.articles = json.load(f)
        
        # 添加哈希值和元数据
        for article in self.articles:
            article_id = article.get('id', '')
            text = article.get('text', '')
            
            # 生成SHA-256哈希值
            text_hash = self._generate_sha256(text)
            article['hash'] = text_hash
            
            # 构建元数据
            metadata = {
                'hash': text_hash,
                'source_institution': source_institution,
                'revision_history': revision_history or [{
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'author': "系统自动生成",
                    'comment': "初始导入"
                }],
                'source_file': os.path.basename(file_path),
                'import_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'content_checksum': text_hash
            }
            
            # 存储元数据
            self.metadata_store[article_id] = metadata
        
        print(f"成功加载 {len(self.articles)} 条法律条款")
        return self.articles
    
    def rule_based_tagging(self) -> List[Dict]:
        """使用基于规则的方法进行初始标签标注"""
        print("正在进行基于规则的初始标签标注...")
        
        # 创建一个新的列表存储带标签的条款
        self.tagged_articles = []
        
        for article in tqdm(self.articles, desc="规则标注进度"):
            tagged_article = article.copy()
            tags = {dimension: set() for dimension in self.tag_trees.keys()}
            
            text = article.get('text', '').lower()
            title = article.get('title', '').lower() if 'title' in article else ''
            full_text = text + " " + title
            
            # 对每个维度应用规则
            for dimension, dimension_rules in self.compiled_rules.items():
                for tag_name, patterns in dimension_rules.items():
                    for pattern in patterns:
                        if pattern.search(full_text):
                            tags[dimension].add(tag_name)
                            
                            # 添加父标签（如果有）
                            parent_tags = self._get_parent_tags(dimension, tag_name)
                            tags[dimension].update(parent_tags)
            
            # 转换集合为列表
            for dim in tags:
                tags[dim] = list(tags[dim])
            
            tagged_article['tags'] = tags
            tagged_article['tag_source'] = 'rule-based'
            self.tagged_articles.append(tagged_article)
        
        print("基于规则的标签标注完成")
        return self.tagged_articles
    
    def _get_parent_tags(self, dimension: str, tag_name: str) -> Set[str]:
        """获取标签的父标签"""
        parent_tags = set()
        
        # 简单的父标签查找逻辑，实际应用中可能需要更复杂的树遍历
        dimension_tree = self.tag_trees.get(dimension, {})
        
        for parent, children in dimension_tree.items():
            if isinstance(children, list) and tag_name in children:
                parent_tags.add(parent)
            elif isinstance(children, dict):
                for _, sub_children in children.items():
                    if tag_name in sub_children:
                        parent_tags.add(parent)
        
        return parent_tags
    
    def prepare_training_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """准备半监督学习的训练数据"""
        print("正在准备半监督学习的训练数据...")
        
        # 提取文本和标签
        texts = [article.get('text', '') for article in self.tagged_articles]
        
        # 为每个维度创建标签矩阵
        dimension_labels = {}
        for dimension in self.tag_trees.keys():
            # 获取所有可能的标签
            all_tags = self._get_all_tags_in_dimension(dimension)
            tag_to_idx = {tag: i for i, tag in enumerate(all_tags)}
            
            # 创建标签矩阵
            labels_matrix = np.zeros((len(texts), len(all_tags)))
            for i, article in enumerate(self.tagged_articles):
                article_tags = article.get('tags', {}).get(dimension, [])
                for tag in article_tags:
                    if tag in tag_to_idx:
                        labels_matrix[i, tag_to_idx[tag]] = 1
            
            dimension_labels[dimension] = labels_matrix
        
        # 训练tokenizer
        self.tokenizer.fit_on_texts(texts)
        
        # 转换文本为序列
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_seq_length, padding='post', truncating='post')
        
        print("训练数据准备完成")
        return padded_sequences, dimension_labels
    
    def _get_all_tags_in_dimension(self, dimension: str) -> List[str]:
        """获取某个维度下的所有标签"""
        all_tags = []
        dimension_tree = self.tag_trees.get(dimension, {})
        
        def traverse_tree(tree_node):
            if isinstance(tree_node, dict):
                for key, value in tree_node.items():
                    all_tags.append(key)
                    traverse_tree(value)
            elif isinstance(tree_node, list):
                all_tags.extend(tree_node)
        
        traverse_tree(dimension_tree)
        return list(set(all_tags))  # 去重
    
    def build_model(self, dimension: str) -> Model:
        """为特定维度构建半监督学习模型"""
        print(f"正在构建 {dimension} 维度的标签模型...")
        
        # 获取该维度的标签数量
        all_tags = self._get_all_tags_in_dimension(dimension)
        num_classes = len(all_tags)
        
        # 构建模型
        inputs = Input(shape=(self.max_seq_length,))
        embedding = Embedding(input_dim=min(len(self.tokenizer.word_index) + 1, 50000),
                              output_dim=self.embedding_dim)(inputs)
        
        # 双向LSTM层
        x = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(embedding)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(self.lstm_units))(x)
        x = Dropout(0.3)(x)
        
        # 输出层（多标签分类）
        outputs = Dense(num_classes, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        print(f"{dimension} 维度模型构建完成")
        return model
    
    def train_models(self, x_train: np.ndarray, y_train_dict: Dict[str, np.ndarray]) -> Dict[str, Model]:
        """训练所有维度的标签模型"""
        print("开始训练所有维度的标签模型...")
        
        # 为每个维度训练一个模型
        for dimension, y_train in y_train_dict.items():
            print(f"训练 {dimension} 维度模型...")
            
            # 构建模型
            model = self.build_model(dimension)
            
            # 创建回调
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            model_checkpoint = ModelCheckpoint(os.path.join(self.script_dir, f'model_{dimension}.keras'), save_best_only=True)
            
            # 训练模型
            model.fit(x_train, y_train,
                     batch_size=self.batch_size,
                     epochs=self.epochs,
                     validation_split=0.2,
                     callbacks=[early_stopping, model_checkpoint],
                     verbose=1)
            
            # 保存模型
            self.models[dimension] = model
        
        print("所有维度模型训练完成")
        return self.models
    
    def semi_supervised_tagging(self, confidence_threshold: float = 0.8) -> List[Dict]:
        """使用半监督学习方法进行标签标注"""
        print("正在进行半监督学习标签标注...")
        
        # 准备训练数据
        x_train, y_train_dict = self.prepare_training_data()
        
        # 训练初始模型
        self.train_models(x_train, y_train_dict)
        
        # 使用训练好的模型进行预测
        updated_articles = []
        for article in tqdm(self.tagged_articles, desc="半监督标注进度"):
            updated_article = article.copy()
            text = article.get('text', '')
            
            # 转换文本为序列
            sequence = self.tokenizer.texts_to_sequences([text])[0]
            padded_sequence = pad_sequences([sequence], maxlen=self.max_seq_length, padding='post', truncating='post')
            
            # 对每个维度进行预测
            for dimension, model in self.models.items():
                # 获取所有可能的标签
                all_tags = self._get_all_tags_in_dimension(dimension)
                
                # 预测
                predictions = model.predict(padded_sequence)[0]
                
                # 应用置信度阈值
                predicted_tags = [all_tags[i] for i, prob in enumerate(predictions) if prob >= confidence_threshold]
                
                # 如果半监督预测结果不为空，则更新标签
                if predicted_tags:
                    # 合并规则标签和半监督标签
                    rule_tags = set(article.get('tags', {}).get(dimension, []))
                    combined_tags = list(rule_tags.union(set(predicted_tags)))
                    updated_article['tags'][dimension] = combined_tags
            
            updated_article['tag_source'] = 'semi-supervised'
            updated_articles.append(updated_article)
        
        self.tagged_articles = updated_articles
        print("半监督学习标签标注完成")
        return self.tagged_articles
    
    def batch_process_articles(self, file_paths: List[str], 
                             source_institution: str = "未知机构") -> Dict[str, List[Dict]]:
        """批量处理多个条款文件"""
        print(f"开始批量处理 {len(file_paths)} 个文件...")
        
        results = {}
        
        for file_path in tqdm(file_paths, desc="文件处理进度"):
            try:
                # 加载文件
                self.load_articles(file_path, source_institution)
                
                # 规则标注
                self.rule_based_tagging()
                
                # 半监督标注
                self.semi_supervised_tagging()
                
                # 保存结果
                results[file_path] = self.tagged_articles.copy()
                
                # 导出结果
                base_name = os.path.splitext(os.path.basename(file_path))[0] + "_标签标注.json"
                output_file = os.path.join(self.script_dir, base_name)
                self.export_tagged_articles(output_file)
                
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")
                results[file_path] = []
        
        print("批量处理完成")
        return results
    
    def parallel_batch_process(self, file_paths: List[str], 
                             source_institution: str = "未知机构") -> Dict[str, List[Dict]]:
        """并行批量处理多个条款文件，适用于百万级数据"""
        print(f"开始并行批量处理 {len(file_paths)} 个文件...")
        
        results = {}
        
        def process_single_file(file_path):
            try:
                # 创建新的实例进行处理，避免线程安全问题
                tagger = LegalMultiDimensionalTagger()
                
                # 加载文件
                tagger.load_articles(file_path, source_institution)
                
                # 规则标注
                tagger.rule_based_tagging()
                
                # 半监督标注
                tagger.semi_supervised_tagging()
                
                # 导出结果
                base_name = os.path.splitext(os.path.basename(file_path))[0] + "_标签标注.json"
                output_file = os.path.join(tagger.script_dir, base_name)
                tagger.export_tagged_articles(output_file)
                
                return file_path, tagger.tagged_articles
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")
                return file_path, []
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {executor.submit(process_single_file, file_path): file_path for file_path in file_paths}
            
            # 收集结果
            for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(file_paths), desc="并行处理进度"):
                file_path = future_to_file[future]
                try:
                    file_path, articles = future.result()
                    results[file_path] = articles
                except Exception as e:
                    print(f"获取文件 {file_path} 结果时出错: {str(e)}")
                    results[file_path] = []
        
        print("并行批量处理完成")
        return results
    
    def export_tagged_articles(self, output_file: str) -> None:
        """导出带标签的条款到JSON文件"""
        print(f"正在导出带标签的条款到: {output_file}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 导出数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.tagged_articles, f, ensure_ascii=False, indent=2)
        
        # 输出完成提示
        print("✅ 多维度标签标注完成！")
        print(f"📄 标注结果已保存至: {output_file}")
        print(f"🏷️  共处理 {len(self.tagged_articles)} 个条款")
    
    def export_metadata(self, output_file: str) -> None:
        """导出元数据到JSON文件"""
        print(f"正在导出元数据到: {output_file}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 导出数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata_store, f, ensure_ascii=False, indent=2)
        
        # 输出元数据完成提示
        print(f"📊 元数据已保存至: {output_file}")
        
        print(f"成功导出 {len(self.metadata_store)} 条元数据")
    
    def analyze_tag_distribution(self) -> pd.DataFrame:
        """分析标签分布情况"""
        print("正在分析标签分布情况...")
        
        # 创建标签计数字典
        tag_counts = {dimension: {} for dimension in self.tag_trees.keys()}
        
        for article in self.tagged_articles:
            article_tags = article.get('tags', {})
            for dimension, tags in article_tags.items():
                for tag in tags:
                    if tag in tag_counts[dimension]:
                        tag_counts[dimension][tag] += 1
                    else:
                        tag_counts[dimension][tag] = 1
        
        # 转换为DataFrame并可视化
        for dimension, counts in tag_counts.items():
            print(f"\n{dimension}维度标签分布:")
            df = pd.DataFrame(list(counts.items()), columns=['标签', '数量'])
            df = df.sort_values('数量', ascending=False)
            
            # 打印前10个标签
            print(df.head(10).to_string(index=False))
            
            # 可视化
            self._visualize_tag_distribution(df, dimension)
        
        return tag_counts
    
    def _visualize_tag_distribution(self, df: pd.DataFrame, dimension: str) -> None:
        """可视化标签分布情况"""
        plt.figure(figsize=(12, 6))
        
        # 如果标签太多，只显示前20个
        if len(df) > 20:
            df = df.head(20)
        
        plt.bar(df['标签'], df['数量'])
        plt.title(f'{dimension}维度标签分布')
        plt.xlabel('标签')
        plt.ylabel('数量')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 保存图片
        output_file = f'tag_distribution_{dimension}.png'
        output_path = os.path.join(self.script_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"标签分布图已保存至: {output_path}")
    
    def evaluate_tagging_results(self, ground_truth_file: str = None) -> Dict[str, Dict[str, float]]:
        """评估标签标注结果（如果有真实标签数据）"""
        if not ground_truth_file or not os.path.exists(ground_truth_file):
            print("未提供真实标签数据，无法进行评估")
            return {}
        
        print(f"正在评估标签标注结果，使用真实标签数据: {ground_truth_file}")
        
        # 加载真实标签数据
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = {article.get('id'): article.get('tags', {}) for article in json.load(f)}
        
        # 创建评估结果字典
        evaluation_results = {}
        
        # 对每个维度进行评估
        for dimension in self.tag_trees.keys():
            # 获取所有可能的标签
            all_tags = self._get_all_tags_in_dimension(dimension)
            
            # 准备真实标签和预测标签
            y_true = []
            y_pred = []
            
            for article in self.tagged_articles:
                article_id = article.get('id')
                
                # 如果有真实标签
                if article_id in ground_truth:
                    # 转换为多标签格式
                    true_tags = set(ground_truth[article_id].get(dimension, []))
                    pred_tags = set(article.get('tags', {}).get(dimension, []))
                    
                    # 对每个标签进行评估
                    for tag in all_tags:
                        y_true.append(1 if tag in true_tags else 0)
                        y_pred.append(1 if tag in pred_tags else 0)
            
            # 计算评估指标（这里简化处理，实际应用中可能需要更复杂的评估）
            if y_true and y_pred:
                # 计算准确率（简单计算）
                correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
                accuracy = correct / len(y_true)
                
                evaluation_results[dimension] = {'accuracy': accuracy}
                print(f"{dimension}维度评估结果: 准确率 = {accuracy:.4f}")
        
        return evaluation_results

# 主函数示例
def main():
    print("📂 正在初始化系统...")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📁 工作目录: {script_dir}")
    
    # 创建标签系统实例
    tagger = LegalMultiDimensionalTagger()
    
    try:
        # 示例1：处理单个文件 - 使用相对路径（segmented_articles 实际文件名）
        input_file = os.path.join(script_dir, "中华人民共和国民法典_条款分割.json")
        if os.path.exists(input_file):
            print(f"\n=== 处理单个文件: {input_file} ===")
            
            # 加载条款
            tagger.load_articles(input_file, source_institution="中国法律法规数据库")
            
            # 规则标注
            tagger.rule_based_tagging()
            
            # 半监督标注
            tagger.semi_supervised_tagging()
            
            # 导出结果 - 使用相对路径
            output_file = os.path.join(script_dir, "tagged_articles.json")
            tagger.export_tagged_articles(output_file)
            
            # 导出元数据 - 使用相对路径
            metadata_file = os.path.join(script_dir, "tagged_metadata.json")
            tagger.export_metadata(metadata_file)
            
            # 分析标签分布
            tagger.analyze_tag_distribution()
            
        # 示例2：批量处理多个文件
        # 查找所有条款分割文件 - 使用脚本目录
        clause_files = [os.path.join(script_dir, f) for f in os.listdir(script_dir) if f.endswith('_条款分割.json')]
        if len(clause_files) > 1:
            print(f"\n=== 批量处理多个文件: {len(clause_files)} 个 ===")
            
            # 对于大量文件，使用并行处理
            if len(clause_files) > 5:
                tagger.parallel_batch_process(clause_files)
            else:
                tagger.batch_process_articles(clause_files)
            
    except Exception as e:
        print(f"程序运行出错: {str(e)}")

if __name__ == "__main__":
    print("🚀 启动多维度标签标注系统...")
    try:
        main()
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()