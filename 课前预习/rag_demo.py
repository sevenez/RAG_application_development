from langchain.document_loaders import TextLoader 
from langchain.text_splitter import CharacterTextSplitter 
from langchain.vectorstores import FAISS 
from langchain.embeddings import HuggingFaceEmbeddings 
# 1. 创建示例文档
with open("demo.txt", "w", encoding="utf-8") as f: 
 f.write(""" 
 人工智能是计算机科学的一个分支。
4 / 5
 机器学习是人工智能的一个子集。
 深度学习是机器学习的一个子集，使用神经网络。
 """) 
# 2. 加载文档
loader = TextLoader("demo.txt", encoding="utf-8") 
documents = loader.load() 
# 3. 文本分块
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0) 
texts = text_splitter.split_documents(documents) 
# 4. 创建向量库
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
db = FAISS.from_documents(texts, embeddings) 
# 5. 简单查询
query = "什么是深度学习？" 
docs = db.similarity_search(query) 
print(f"查询: {query}") 
print("找到的相关内容:") 
for doc in docs: 
 print(doc.page_content)