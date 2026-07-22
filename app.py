import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG 智能问答助手", layout="centered")
st.title("📚 智能问答助手")

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    st.error("❌ 未找到 API Key")
    st.stop()

embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=api_key)
llm = ChatTongyi(model="qwen-max", temperature=0.1, dashscope_api_key=api_key)

# ===== 深度分析 Prompt =====
prompt_template = """你是一个专业的学术文档分析助手。请严格基于以下【上下文】信息回答用户的问题。

【回答要求】：
1. 根据文档类型，自动选择合适的深度分析结构：
   - 实验报告：目的、方法、数据、结论、局限性
   - 学术论文：背景、方法、发现、贡献、未来工作
   - 技术文档：功能、步骤、参数、注意事项
2. 保留代码、数据、公式。
3. 缺失信息请说明。
4. 不要编造。

【上下文】：
{context}

【问题】：
{question}

【回答】："""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

uploaded_files = st.file_uploader(
    "上传文档（PDF / TXT，可多选）",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key="uploader"
)

def load_document(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("不支持的文件类型")
    return loader.load()

# 初始化 session_state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if uploaded_files and not st.session_state.uploaded:
    current_files = sorted([f.name for f in uploaded_files])
    current_key = "_".join(current_files)
    if st.session_state.get("last_files_key") != current_key:
        st.session_state["cleaned"] = False
        st.session_state["last_files_key"] = current_key

    if os.path.exists("./chroma_db") and not st.session_state.get("cleaned", False):
        try:
            shutil.rmtree("./chroma_db")
            st.info("清理旧数据...")
            st.session_state["cleaned"] = True
        except PermissionError:
            st.warning("旧数据正在使用，请重启后重试。")
            st.stop()

    all_chunks = []
    total_pages = 0
    for uploaded_file in uploaded_files:
        file_path = f"./temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        file_ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
        with st.spinner(f"处理 {uploaded_file.name}..."):
            docs = load_document(file_path, file_ext)
            total_pages += len(docs)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
            )
            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata["file_name"] = uploaded_file.name
            all_chunks.extend(chunks)

    st.write(f"📄 共处理 {len(uploaded_files)} 个文件，{total_pages} 页，{len(all_chunks)} 个文本块")

    with st.spinner("构建索引..."):
        vectordb = Chroma.from_documents(documents=all_chunks, embedding=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 30})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        st.session_state.qa_chain = qa_chain
        st.session_state.retriever = retriever
        st.session_state.uploaded = True

    st.success(f"✅ {len(uploaded_files)} 个文档已就绪")

# 提问
if st.session_state.qa_chain:
    question = st.text_input("请输入你的问题：")
    if st.button("提问"):
        if question:
            with st.spinner("思考中..."):
                result = st.session_state.qa_chain.invoke({"query": question})
                answer = result["result"]
                source_docs = result.get("source_documents", [])
                st.write("**回答：**", answer)
                if source_docs:
                    st.write("---")
                    st.write("📖 **引用来源：**")
                    seen = set()
                    for doc in source_docs:
                        fname = doc.metadata.get("file_name", "未知文件")
                        page = doc.metadata.get("page", "?")
                        snippet = doc.page_content[:100].replace("\n", " ")
                        key = f"{fname}_{page}"
                        if key not in seen:
                            seen.add(key)
                            st.write(f"- {fname} (第{page}页): {snippet}...")
