import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 加载 .env 文件
load_dotenv()

st.set_page_config(page_title="RAG 智能问答助手", layout="centered")
st.title("📚 智能问答助手")

# ===== 1. 配置 API Key（仅从环境变量读取）=====
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    st.error("❌ 未找到 API Key，请在项目根目录创建 .env 文件并设置 DASHSCOPE_API_KEY")
    st.stop()

# ===== 2. 初始化模型 =====
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=api_key
)

llm = ChatTongyi(
    model="qwen-max",
    temperature=0.1,
    dashscope_api_key=api_key
)

# ===== 3. 自定义 Prompt =====
prompt_template = """你是一个严谨的实验报告分析助手。请严格基于以下【上下文】信息回答用户的问题。

【回答要求】：
1. 如果用户问"总结"、"介绍"或"概述"，请按以下结构分点回答，不要遗漏任何要点：
   - 实验目的/目标
   - 实验要求/任务列表（保留编号）
   - 关键技术点
   - 实验结果或提交要求
2. 如果文档中包含代码、填空或具体数值，请在回答中尽量保留这些细节。
3. 如果【上下文】中的信息不完整，请说明"根据现有文档，以下信息未提及："并列出缺失项。
4. 绝对不要使用你自己的知识补充文档中没有的内容。

【上下文】：
{context}

【问题】：
{question}

【回答】："""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ===== 4. 文件上传与处理 =====
uploaded_file = st.file_uploader("上传你的 PDF 文档", type="pdf")

if uploaded_file is not None:
    # 自动清理旧向量库
    current_file = uploaded_file.name
    if st.session_state.get("last_file") != current_file:
        st.session_state["cleaned"] = False
        st.session_state["last_file"] = current_file

    if os.path.exists("./chroma_db") and not st.session_state.get("cleaned", False):
        try:
            shutil.rmtree("./chroma_db")
            st.info("🔄 已清理旧文档数据，正在处理新文档...")
            st.session_state["cleaned"] = True
        except PermissionError:
            st.warning("⚠️ 旧数据正在使用中，请关闭其他程序或重启后重试。")
            st.stop()

    # 保存上传的文件
    pdf_path = f"./temp_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("正在处理文档..."):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)

        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        retriever = vectordb.as_retriever(search_kwargs={"k": 20})

        # 创建问答链
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        qa_chain.combine_documents_chain.llm_chain.prompt = PROMPT

        st.success("✅ 文档处理完成，可以提问了！")

        # 调试面板
        # with st.expander("🔍 调试：查看检索到的文本块"):
        #     sample_q = "实验目的 实验要求"
        #     retrieved_docs = retriever.get_relevant_documents(sample_q)
        #     st.write(f"共检索到 {len(retrieved_docs)} 个文本块：")
        #     for i, doc in enumerate(retrieved_docs):
        #         st.write(f"--- 块 {i + 1} ---")
        #         st.write(doc.page_content[:400])
        #         st.write("---")

        # 提问与回答
        question = st.text_input("请输入你的问题：")
        if question:
            with st.spinner("思考中..."):
                result = qa_chain.invoke({"query": question})
                answer = result["result"]
                source_docs = result.get("source_documents", [])

                st.write("**回答：**", answer)

                if source_docs:
                    st.markdown("---")
                    st.markdown("📖 **引用来源：**")
                    seen = set()
                    for doc in source_docs:
                        page = doc.metadata.get("page", "未知页码")
                        snippet = doc.page_content[:150].replace("\n", " ")
                        if snippet not in seen:
                            seen.add(snippet)
                            st.markdown(f"- 第{page}页：{snippet}...")