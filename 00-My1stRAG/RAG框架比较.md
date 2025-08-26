# RAG框架比较

本文档对比了五种不同的RAG（检索增强生成）实现方法，它们都使用了DeepSeek大语言模型和中文法律文档（《中华人民共和国民营经济促进法》）作为数据源，并回答相同的问题："哪些部门负责促进民营经济发展的工作？"

## 1. LlamaIndex实现 (01_LlamaIndex_DeepSeek.py)

**功能特点：**
- 使用LlamaIndex框架构建RAG系统
- 采用HuggingFace的bge-small-zh作为嵌入模型
- 直接从文件加载文档，无需手动分块
- 自动构建向量索引并创建查询引擎
- 实现了简单直接的问答功能

**技术特点：**
- 代码结构最为简洁，仅约30行有效代码
- 使用SimpleDirectoryReader直接加载文档
- 框架内部自动处理文档分块和索引构建
- 使用VectorStoreIndex作为检索基础
- 通过as_query_engine()方法快速创建问答引擎
- 无需手动构建提示模板，框架内部处理

**优势：**
- 代码简洁，实现最为精简
- 自动处理文档加载、索引构建等步骤
- 适合快速构建原型系统
- 对初学者友好，入门门槛低

## 2. LangChain基础实现 (02_LangChain_DeepSeek_v1.py)

**功能特点：**
- 使用LangChain框架构建RAG系统
- 手动实现文档加载、分块、嵌入和检索流程
- 使用RecursiveCharacterTextSplitter进行文档分块
- 采用InMemoryVectorStore作为向量存储
- 通过提示模板构建查询

**技术特点：**
- 流程分解为明确的步骤，每个步骤独立实现
- 使用TextLoader加载文档
- 明确设置文档分块参数(chunk_size=1000, chunk_overlap=200)
- 使用HuggingFaceEmbeddings生成文本向量
- 通过similarity_search方法检索相关文档
- 手动构建提示模板和上下文拼接
- 直接调用LLM生成答案

**优势：**
- 流程清晰，各步骤分离
- 可以灵活调整每个环节的参数
- 适合理解RAG的基本工作原理
- 提供了RAG流程的完整视图

## 3. LangChain LCEL实现 (03_LangChain_LCEL_DeepSeek_v1.py)

**功能特点：**
- 使用LangChain表达式语言(LCEL)构建RAG系统
- 采用声明式链式结构组织RAG流程
- 使用RunnablePassthrough实现数据流转
- 保持了与基础实现相同的文档处理和检索方式
- 增加了输出解析器

**技术特点：**
- 使用LCEL（LangChain表达式语言）的管道式语法
- 将检索器封装为可调用对象(retriever)
- 使用RunnablePassthrough传递问题
- 通过字典结构组织上下文和问题
- 使用StrOutputParser处理模型输出
- 整个RAG流程被组织为一个声明式链条
- 通过chain.invoke()方法执行整个流程

**优势：**
- 代码结构更加模块化
- 使用链式API，更易于扩展和修改
- 适合构建复杂的多步骤工作流
- 更好的可维护性和可测试性
- 流程可视化更清晰

## 4. LangGraph实现 (04_LangGraph_DeepSeek.py)

**功能特点：**
- 使用LangGraph框架构建RAG系统
- 基于状态图的工作流设计
- 定义了明确的状态类型和转换函数
- 将检索和生成定义为独立步骤
- 使用hub获取标准RAG提示模板

**技术特点：**
- 使用StateGraph构建基于状态的工作流
- 定义了明确的State类型（TypedDict）
- 将检索(retrieve)和生成(generate)定义为独立函数
- 使用LangChain Hub获取标准RAG提示模板
- 通过add_sequence方法定义工作流顺序
- 使用compile()方法编译状态图
- 禁用了LangSmith跟踪功能
- 通过graph.invoke()执行整个工作流

**优势：**
- 支持更复杂的工作流和状态管理
- 适合构建多步骤、可能包含循环的复杂应用
- 便于扩展为具有反馈循环的高级RAG系统
- 更好地支持条件分支和决策点
- 适合构建具有记忆和状态的对话系统

## 5. 从零构建实现 (05_Scratch_DeepSeek.py)

**功能特点：**
- 不依赖RAG框架，从零构建完整流程
- 使用sentence-transformers生成嵌入
- 采用FAISS库进行高效向量检索
- 手动构建提示词和上下文
- 直接调用DeepSeek API生成回答

**技术特点：**
- 使用正则表达式手动分割文档(re.split)
- 采用sentence-transformers库生成文本嵌入
- 使用FAISS库构建向量索引和检索
- 手动构建包含引用编号的提示词
- 通过OpenAI客户端直接调用DeepSeek API
- 使用自定义的base_url指向DeepSeek服务
- 完全自主控制检索和生成过程
- 错误处理更加明确

**优势：**
- 完全控制实现的每个细节
- 无框架依赖，更灵活的定制能力
- 适合理解RAG的底层工作原理
- 可以针对特定需求进行深度优化
- 更低的依赖性，减少版本兼容问题
- 便于集成到现有系统中

## 总结

这五种实现方式展示了构建RAG系统的不同方法，从高级框架到底层实现：

### 框架对比

| 实现方式 | 代码复杂度 | 灵活性 | 适用场景 | 特点 |
|---------|-----------|-------|---------|------|
| LlamaIndex | 低 | 中 | 快速原型开发 | 最简洁，自动化程度高 |
| LangChain基础 | 中 | 中 | 学习RAG原理 | 流程清晰，步骤分明 |
| LangChain LCEL | 中 | 高 | 复杂工作流 | 模块化设计，链式结构 |
| LangGraph | 高 | 高 | 复杂交互系统 | 基于状态图，支持复杂流程 |
| 从零构建 | 高 | 最高 | 深度定制系统 | 完全控制，无框架限制 |

### 技术栈对比

| 实现方式 | 文档加载 | 文档分块 | 向量存储 | 检索方式 | 提示词构建 |
|---------|---------|---------|---------|---------|----------|
| LlamaIndex | SimpleDirectoryReader | 自动 | VectorStoreIndex | 自动 | 自动 |
| LangChain基础 | TextLoader | RecursiveCharacterTextSplitter | InMemoryVectorStore | similarity_search | 手动模板 |
| LangChain LCEL | TextLoader | RecursiveCharacterTextSplitter | InMemoryVectorStore | retriever | 手动模板+链式 |
| LangGraph | TextLoader | RecursiveCharacterTextSplitter | InMemoryVectorStore | 函数封装 | hub.pull模板 |
| 从零构建 | 直接读取 | 正则表达式 | FAISS | 自定义检索 | 完全手动 |

选择哪种实现方式取决于项目需求、复杂度和定制化程度。对于简单应用，LlamaIndex或基础LangChain可能足够；对于复杂系统，LangGraph或从零构建可能更合适。随着应用复杂度的增加，从LlamaIndex到从零构建的实现方式提供了越来越多的控制能力和灵活性。
