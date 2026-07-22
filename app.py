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

# 深度分析 Prompt（与之前相同）
prompt_template = """你是一个专业的学术文档分析助手。请严格基于以下【上下文】信息回答用户的问题。

【回答要求】：
1. 根据文档类型，自动选择合适的深度分析结构：

   **如果是实验报告**：
   - 实验目的/研究目标
   - 实验方法/技术路线（包括使用的工具、参数设置）
   - 实验数据与结果分析（包括数值、图表趋势）
   - 结论与意义
   - 局限性或改进方向（如文档中有提及）

   **如果是学术论文**：
   - 研究背景与动机（为什么做这个研究）
   - 研究方法/模型架构（用了什么方法、数据来源）
   - 主要发现/实验结果（关键数据、对比基准）
   - 结论与贡献（解决了什么问题）
   - 未来工作或局限性（文档中提及的不足）

   **如果是技术文档/手册**：
   - 功能概述（解决什么问题）
   - 使用步骤/操作流程（按顺序列出）
   - 关键参数/配置选项
   - 常见问题/注意事项

2. 如果文档中包含代码、数据表格、数学公式，请在回答中尽量保留并解释。
3. 如果文档中有对比实验或数据分析，请提取对比结论和关键趋势。
4. 如果【上下文】中的信息不完整，请明确说明哪些内容在文档中未提及。
5. 绝对不要使用你自己的知识补充文档中没有的内容。
6. 回答时使用分点或编号，逻辑层次清晰。

【上下文】：
{context}

【问题】：
{question}

【回答】："""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ===== 文件上传（支持 PDF 和 TXT） =====
uploaded_files = st.file_uploader(
    "上传你的文档（PDF / TXT，可多选）",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key="file_uploader"
)

def load_document(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件类型: {file_type}")
    return loader.load()

if uploaded_files:
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
        st.info(f"正在处理: {uploaded_file.name} ({file_ext})")

        with st.spinner(f"正在处理: {uploaded_file.name}..."):
            docs = load_document(file_path, file_ext)
            total_pages += len(docs)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
            )
            chunks = text_splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata["file_name"] = uploaded_file.name
            all_chunks.extend(chunks)

    st.write(f"📄 共处理 {len(uploaded_files)} 个文件，{total_pages} 页（总块数: {len(all_chunks)}）")

    with st.spinner("正在构建向量索引..."):
        vectordb = Chroma.from_documents(documents=all_chunks, embedding=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 30})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    st.success(f"✅ {len(uploaded_files)} 个文档已就绪，可以提问了！")

    # 调试面板
    with st.expander("🔍 调试：查看检索到的文本块"):
        sample_q = "总结 介绍"
        try:
            retrieved_docs = retriever.get_relevant_documents(sample_q)
            st.write(f"共检索到 {len(retrieved_docs)} 个文本块：")
            for i, doc in enumerate(retrieved_docs):
                file_name = doc.metadata.get("file_name", "未知文件")
                st.write(f"--- 块 {i+1} (来自: {file_name}) ---")
                st.write(doc.page_content[:400])
                st.write("---")
        except Exception as e:
            st.warning(f"调试失败: {e}")

    # 提问表单
    with st.form(key="qa_form"):
        question = st.text_input("请输入你的问题：")
        submitted = st.form_submit_button("提问")

    if submitted and question:
        with st.spinner("思考中..."):
            result = qa_chain.invoke({"query": question})
            answer = result["result"]
            source_docs = result.get("source_documents", [])

            st.write("**回答：**", answer)

            # ===== 显示引用来源 =====
            if source_docs:
                st.markdown("---")
                st.markdown("📖 **引用来源：**")
                seen = set()
                for doc in source_docs:
                    file_name = doc.metadata.get("file_name", "未知文件")
                    page = doc.metadata.get("page", "未知页码")
                    snippet = doc.page_content[:150].replace("\n", " ")
                    key = f"{file_name}_{page}"
                    if key not in seen:
                        seen.add(key)
                        st.markdown(f"- **文件**: {file_name}，**第{page}页**：{snippet}...")
