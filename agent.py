import sys
sys.path.insert(0, ".")

import os
from dotenv import load_dotenv
import dashscope
from dashscope import Generation, TextEmbedding
from tools import search_documents, extract_tables, extract_images, read_image
from pypdf import PdfReader

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def get_embedding(text):
    resp = TextEmbedding.call(model="text-embedding-v1", input=text)
    if resp.status_code == 200:
        return resp.output["embeddings"][0]["embedding"]
    return None

def write_file(content, filename="output.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已保存到 {filename}"

def simple_agent(question, chunks, embeddings, history):
    # ===== 把历史对话拼成一段文字 =====
    history_text = ""
    for turn in history:
        history_text += f"用户：{turn['user']}\n助手：{turn['assistant']}\n\n"

    # ===== 第一步：判断应该调用哪个工具 =====
    decision_prompt = f"""你是一个智能助手。以下是之前的对话历史：
{history_text}
用户最新问题：{question}
你有两个工具：
1. search_documents：从文档中检索相关信息。
2. write_file：把内容保存到文件。
请判断：要回答这个问题，应该调用哪个工具？还是直接回答？
如果需要调用工具，请只回复工具名字（search_documents 或 write_file）；
如果不需要，请回复"直接回答"。"""

    decision = Generation.call(
        model="qwen-max",
        prompt=decision_prompt,
        temperature=0.1
    )
    decision_text = decision.output["text"]

    # ===== 第二步：根据判断执行操作 =====
    if "search_documents" in decision_text:
        print("→ Agent 决定：调用 search_documents 工具")
        retrieved = search_documents(question, chunks, embeddings)
        context = "\n\n".join(retrieved)

    elif "write_file" in decision_text:
        print("→ Agent 决定：调用 write_file 工具")
        # 保存上一轮的回答，如果没有历史，就保存当前问题
        if history:
            content_to_save = history[-1]["assistant"]
        else:
            content_to_save = question
        write_file(content_to_save, "agent_output.txt")
        context = f"已保存到 agent_output.txt"

    else:
        print("→ Agent 决定：直接回答")
        context = "（无检索结果）"

    # ===== 第三步：生成最终回答 =====
    final_prompt = f"""请基于以下【上下文】和之前的对话历史回答用户的问题。
如果上下文没有相关信息，请如实说"未找到"，不要编造。

【之前的对话】：
{history_text}

【上下文】：
{context}

【问题】：
{question}

【回答】："""

    final_answer = Generation.call(
        model="qwen-max",
        prompt=final_prompt,
        temperature=0.1
    )
    return final_answer.output["text"]


if __name__ == "__main__":
    pdf_path = "test.pdf"

    print("1. 正在读取PDF纯文本...")
    text = load_pdf(pdf_path)

    print("2. 正在提取表格...")
    tables = extract_tables(pdf_path)
    text = text + "\n\n【表格内容】\n" + tables

    print("3. 正在提取并识别图片（可能需要几十秒，请稍候）...")
    images = extract_images(pdf_path)
    image_texts = []
    for img_path in images:
        print(f"正在识别图片: {img_path}")
        img_desc = read_image(img_path)
        image_texts.append(img_desc)

    # 把图片的描述也拼进总文本里
    text = text + "\n\n【图片内容】\n" + "\n".join(image_texts)

    print(f"✅ 文档解析完成！总字符数：{len(text)}")

    # 接下来是切分和向量化（和你原来的一样）
    chunks = split_text(text)
    print(f"切分成 {len(chunks)} 个块")

    print("正在生成向量...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        if emb is None:
            print(f"第 {i + 1} 个块向量化失败")
            exit()
        embeddings.append(emb)
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{len(chunks)} 个块")

    print("向量生成完成！")

    # 对话循环（和你原来的一样）
    history = []
    while True:
        question = input("\n请输入你的问题（输入 q 退出）：")
        if question.lower() == "q":
            break

        answer = simple_agent(question, chunks, embeddings, history)
        print("\n最终回答：")
        print(answer)
        history.append({"user": question, "assistant": answer})