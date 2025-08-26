# 1. 导入必要的库
import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量
print("正在尝试加载.env文件...")
load_dotenv(verbose=True)  # verbose=True会打印加载过程的详细信息

from langchain_community.document_loaders import TextLoader  # 用于加载本地文本文件
# 从本地加载法律文档
loader = TextLoader(
    file_path="../20-Data/法律文档/中华人民共和国民营经济促进法.txt",
    encoding='utf-8'  # 指定编码，避免中文乱码问题
)
docs = loader.load()

# 2. 分割文档
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# 3. 设置嵌入模型（保持使用本地模型，无需API）
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 4. 创建向量存储
from langchain_core.vectorstores import InMemoryVectorStore

vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(all_splits)

# 5. 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6. 创建提示模板
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
基于以下上下文，回答问题。如果上下文中没有相关信息，
请说"我无法从提供的上下文中找到相关信息"。
上下文: {context}
问题: {question}
回答:""")

# 7. 设置语言模型（和输出解析器
from langchain_deepseek import ChatDeepSeek  # 导入DeepSeek适配器
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 从系统环境变量获取API密钥
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    print("错误：无法从系统环境变量获取DEEPSEEK_API_KEY")
    print("请确保已设置DEEPSEEK_API_KEY环境变量")
    exit(1)  # 如果没有API密钥，则退出程序
else:
    print(f"成功获取API密钥: {api_key[:5]}...{api_key[-4:]}")

# 初始化DeepSeek模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=2048,
    api_key=api_key  # 使用验证过的API密钥
)

# 8. 构建 LCEL 链（保持链结构不变，仅替换了LLM）
chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 9. 执行查询（问题改为与法律文档相关）
question = "哪些部门负责促进民营经济发展的工作？"
response = chain.invoke(question)
print(response)
