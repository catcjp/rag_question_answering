import streamlit as st
import os
import shutil
from dotenv import load_dotenv
import pypdf
from pypdf import PdfReader
import dashscope
from dashscope import TextEmbedding
from dashscope import Generation
import json
import hashlib

load_dotenv()

st.set_page_config(page_title="RAG 智能问答助手", layout="centered")
st.title("📚 智能问答助手")

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    st.error("❌ 未找到 API Key")
    st.stop()

dashscope.api_key = api_key

# 配置
EMBEDDING_MODEL = "text-embedding-v1"
LLM_MODEL = "qwen-max"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5

# ===== 文本切分函数 =====
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

# ===== 获取向量 =====
def get_embedding(text):
    resp = TextEmbedding.call(
        model=EMBEDDING_MODEL,
        input=text
    )
    if resp.status_code == 200:
        return resp.output["embeddings"][0]["embedding"]
    else:
        st.error(f"向量化失败: {resp.message}")
        return None

# ===== 向量检索（简单余弦相似度）=====
def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = sum(x*x for x in a) ** 0.5
    norm_b = sum(y*y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)

def retrieve(query_embedding, chunks, embeddings, top_k=TOP_K):
    scores = []
    for i, emb in enumerate(embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((score, i))
    scores.sort(reverse=True)
    top_indices = [idx for _, idx in scores[:top_k]]
    return [chunks[i] for i in top_indices]

# ===== 调用大模型生成回答 =====
def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""你是一个专业的学术文档分析助手。请严格基于以下【上下文】信息回答用户的问题。

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
    
    resp = Generation.call(
        model=LLM_MODEL,
        prompt=prompt,
        temperature=0.1
    )
    if resp.status_code == 200:
        return resp.output["text"]
    else:
        return f"生成失败: {resp.message}"

# ===== UI =====
uploaded_files = st.file_uploader(
    "上传文档（PDF / TXT，可多选）",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key="uploader"
)

def extract_text(file_path, file_type):
    if file_type == "pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    elif file_type == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return ""

# 初始化 session_state
if "vector_data" not in st.session_state:
    st.session_state.vector_data = None  # 存储 (chunks, embeddings, file_names)
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if uploaded_files and not st.session_state.uploaded:
    all_chunks = []
    all_embeddings = []
    all_file_names = []
    total_chars = 0
    
    for uploaded_file in uploaded_files:
        file_path = f"./temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        file_ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
        
        with st.spinner(f"处理 {uploaded_file.name}..."):
            text = extract_text(file_path, file_ext)
            chunks = split_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_file_names.append(uploaded_file.name)
            total_chars += len(text)
            st.write(f"📄 {uploaded_file.name}: {len(text)} 字符 → {len(chunks)} 个块")
    
    st.write(f"📊 共 {len(all_chunks)} 个文本块，{total_chars} 字符")
    
    with st.spinner("正在生成向量索引..."):
        for i, chunk in enumerate(all_chunks):
            emb = get_embedding(chunk)
            if emb is None:
                st.error("向量化失败，请检查 API Key")
                st.stop()
            all_embeddings.append(emb)
            if (i + 1) % 10 == 0:
                st.write(f"已处理 {i+1}/{len(all_chunks)} 个块...")
    
    st.session_state.vector_data = {
        "chunks": all_chunks,
        "embeddings": all_embeddings,
        "file_names": all_file_names
    }
    st.session_state.uploaded = True
    st.success(f"✅ {len(uploaded_files)} 个文档已就绪")

# ===== 提问 =====
if st.session_state.vector_data:
    question = st.text_input("请输入你的问题：")
    if st.button("提问"):
        if question:
            with st.spinner("思考中..."):
                data = st.session_state.vector_data
                # 获取问题向量
                q_emb = get_embedding(question)
                if q_emb is None:
                    st.error("问题向量化失败")
                else:
                    # 检索
                    retrieved_chunks = retrieve(q_emb, data["chunks"], data["embeddings"])
                    # 生成回答
                    answer = generate_answer(question, retrieved_chunks)
                    st.write("**回答：**", answer)
                    
                    # 引用来源
                    st.write("---")
                    st.write("📖 **引用来源：**")
                    seen = set()
                    for i, chunk in enumerate(retrieved_chunks):
                        fname = data["file_names"][data["chunks"].index(chunk)]
                        snippet = chunk[:100].replace("\n", " ")
                        key = f"{fname}_{snippet[:20]}"
                        if key not in seen:
                            seen.add(key)
                            st.write(f"- {fname}: {snippet}...")
