# 📚 RAG 智能文档问答助手

基于 RAG（Retrieval-Augmented Generation）架构的智能文档问答系统，支持上传 PDF/TXT 文档，通过向量检索与大模型生成实现精准问答。

🔗 **在线演示**：[点击试用](https://ragquestionanswering-gmwcribvq4ez7htvtz3x2s.streamlit.app/)

## ✨ 功能特性

- 📄 **多格式支持**：PDF / TXT 文档上传，可同时上传多个文件
- 🔍 **智能检索**：基于向量相似度检索，快速定位相关段落
- 🤖 **深度回答**：自动识别文档类型（实验报告/论文/技术文档），按结构输出回答
- 📖 **引用来源**：回答附带文档来源，结果可追溯
- 🚀 **开箱即用**：无需本地部署，浏览器即可访问

## 🛠️ 技术栈

| 组件 | 技术选型 |
|------|----------|
| 前端界面 | Streamlit |
| 大语言模型 | 通义千问 qwen-max |
| 嵌入模型 | text-embedding-v1 |
| 文档解析 | pypdf |
| 向量检索 | 自定义余弦相似度 |
| 部署平台 | Streamlit Cloud |

## 📁 项目结构

```
rag_question_answering/
├── app.py              # 主程序
├── requirements.txt    # 依赖清单
├── runtime.txt         # Python 版本指定（3.11）
└── README.md           # 项目说明
```

## 🚀 本地运行

```bash
# 1. 克隆仓库
git clone https://github.com/catcjp/rag_question_answering.git
cd rag_question_answering

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
# 在项目根目录创建 .env 文件，写入：
# DASHSCOPE_API_KEY=你的密钥

# 4. 启动应用
streamlit run app.py
```

## 📊 效果展示

| 功能 | 示例 |
|------|------|
| 多文件上传 | 支持同时上传 PDF/TXT |
| 结构化总结 | 自动提取实验目的/方法/数据/结论 |
| 引用来源 | 回答末尾标注文档来源 |

## 📝 待优化方向

- [ ] 支持 DOCX 格式
- [ ] 增加对话记忆功能
- [ ] 支持图片/扫描件 OCR 识别
- [ ] 自定义分块参数调节

## 📄 License

MIT License

## 👤 作者

GitHub: [@catcjp](https://github.com/catcjp)
