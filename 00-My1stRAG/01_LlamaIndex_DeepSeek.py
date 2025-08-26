# 1：准备环境
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek
from dotenv import load_dotenv
import os

# 从系统环境变量获取API密钥
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    print("错误：无法从系统环境变量获取DEEPSEEK_API_KEY")
    print("请确保已设置DEEPSEEK_API_KEY环境变量")
    exit(1)
else:
    print(f"成功获取API密钥: {api_key[:5]}...{api_key[-4:]}")

# 加载本地嵌入模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")
# 创建 Deepseek LLM
llm = DeepSeek(
    model="deepseek-chat",
    api_key=api_key
)

# 2：加载数据
documents = SimpleDirectoryReader(input_files=["../20-Data/法律文档/中华人民共和国民营经济促进法.txt"]).load_data()

# 3：构建索引
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# 4：创建问答引擎
query_engine = index.as_query_engine(
    llm=llm
)

# 5: 开始问答
print(query_engine.query("哪些部门负责促进民营经济发展的工作?"))
