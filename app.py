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

load_dotenv()

st.set_page_config(page_title="RAG 智能问答助手", layout="centered")
st.title("📚 智能问答助手")

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    st.error("❌ 未找到 API Key")
    st.stop()

embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=api_key)
llm = ChatTongyi(model="qwen-max", temperature=0.1, dashscope_api_key=api_key)

prompt_template = """你是一个严谨的实验报告分析助手。请严格基于以下【上下文】信息回答用户的问题。

【回答要求】：
1. 如果用户问"总结"、"介绍"或"概述"，请按以下结构分点回答：
   - 实验目的/目标
   - 实验要求/任务列表（保留编号）
   - 关键技术点
   - 实验结果或提交要求
2. 如果文档中包含代码或具体数值，请在回答中尽量保留这些细节。
3. 如果【上下文】中的信息不完整，请说明哪些内容在文档中未提及。
4. 绝对不要使用你自己的知识补充文档中没有的内容。

【上下文】：
{context}

【问题】：
{question}

【回答】："""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

uploaded_file = st.file_uploader("上传你的 PDF 文档", type="pdf", key="file_uploader")

if uploaded_file is not None:
    # 清理旧向量库（与之前相同）
    if st.session_state.get("last_file") != uploaded_file.name:
        st.session_state["cleaned"] = False
        st.session_state["last_file"] = uploaded_file.name

    if os.path.exists("./chroma_db") and not st.session_state.get("cleaned", False):
        try:
            shutil.rmtree("./chroma_db")
            st.info("清理旧数据...")
            st.session_state["cleaned"] = True
        except PermissionError:
            st.warning("旧数据正在使用，请重启后重试。")
            st.stop()

    pdf_path = f"./temp_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("正在处理..."):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
)
        chunks = text_splitter.split_documents(docs)
        vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 15})
        qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
        st.success("✅ 文档已就绪，可以提问了！")

    # 使用 st.form 稳定交互
    with st.form(key="qa_form"):
        question = st.text_input("请输入你的问题：")
        submitted = st.form_submit_button("提问")

    if submitted and question:
        with st.spinner("思考中..."):
            result = qa_chain.invoke({"query": question})
            answer = result["result"]
            st.write("**回答：**", answer)
