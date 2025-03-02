# 🚀 **DeepSeek 私人 Chatbot 1.0：一体化集成 GraphRAG & 聊天记录 & 网络搜索**
**(100% 免费，私人，无需联网，私有化部署)**  

## 介绍

🔥 **DeepSeek + NOMIC + FAISS + Neural Reranking + HyDE + GraphRAG + Chat Memory + searxng web search  = 终极 RAG 技术栈!**  

本聊天机器人通过最小化集成 **DeepSeek-8B**，**qwen2-1.5b 文本嵌入模型**，**BAAI/bge 重排模型(交叉编码器)**，**GraphRAG**，和 **聊天历史**  来支持从 PDFs，DOCX，and TXT 文件快速、精确、可解释的检索增强 LLM 服务。只有一个 ``app.py`` 文件，灵活易拓展，适合作为你的 LLM chatbot 学习起点或定制个性化机器人。 

如果你厌倦了 Dify 以及各种 xxx studio 的 all-in-one 的庞杂和配置繁琐，只需要知识库、在线搜索、对话历史等基本功能，私有化部署在自己的 Home Lab 里，随时可调整控制每个细节，那这个项目就是为你准备的。

---

# **安装 & 设置**

你可以选择以下两种安装方式之一来部署 **DeepSeek Private Chatbot**：

1。**传统 (Python/venv) 安装**  
2。**Docker 容器化安装** (非常理想的容器化部署方式)

---

## **1️⃣ 传统 (Python/venv) 安装**

### **步骤 A: Clone the Repository & Install Dependencies**
```
git clone https://github.com/faywong/DeepSeek-Private-Chatbot.git
cd DeepSeek-Private-Chatbot

# Create a virtual environment
python -m venv venv

# Activate your environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip (optional，but recommended)
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

### **步骤 B: 下载 & 设置 Ollama**
1。**下载 Ollama** → [https://ollama.com/](https://ollama.com/)  
2。**拉取所需模型**:
   ```
   ollama pull deepseek-r1:8b
   ollama pull rjmalagon/gte-qwen2-1.5b-instruct-embed-f16
   ```
   *注意: 你可以通过编辑 `MODEL` 或 `EMBEDDINGS_MODEL` 环境变量（或 `.env` 文件）来使用其他的模型*   

### **步骤 C: 运行对话机器人**
1。启动 **Ollama** 服务:
   ```
   ollama serve
   ```
2。启动 app.py（Streamlit 应用）:
   ```
   streamlit run app.py
   ```
3。打开浏览器访问 **[http://localhost:8501](http://localhost:8501)** 来访问 chatbot UI.

---

## **2️⃣ Docker 容器化安装**

### **A) 单容器实例方式 (Ollama 运行在你的宿主机)**

如果 **Ollama** 已经运行在你的宿主机并侦听在 `localhost:11434`，那么可以直接:

1。**构建 & 运行**:
   ```
   docker-compose build
   docker-compose up
   ```
2。对话机器人就运行在 **[http://localhost:8501](http://localhost:8501)**。

### **B) 两容器实例编排方式 (Ollama 运行在 docker 里)**

如果你喜欢 **一切** 在 Docker，那么可以通过 `docker compose` 来编排下:
```
version: "3.8"

services:
  ollama:
    image: ghcr.io/jmorganca/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"

  searxng:
    image: docker.io/searxng/searxng:latest
    container_name: searxng
    ports:
      - "4000:8080"
    volumes:
      - ./searxng:/etc/searxng
    restart: unless-stopped

  deepgraph-rag-service:
    container_name: deepgraph-rag-service
    build: .
    ports:
      - "8501:8501"
    environment:
      - OLLAMA_API_URL=http://ollama:11434
      - MODEL=deepseek-r1:8b
      - EMBEDDINGS_MODEL=rjmalagon/gte-qwen2-1.5b-instruct-embed-f16:latest
      - CROSS_ENCODER_MODEL=BAAI/bge-reranker-large
    depends_on:
      - ollama
      - searxng

```

然后构建和启动:
```
docker-compose build
docker-compose up
```

**Ollama** 和聊天机器人、搜索服务 `searxng` 都在容器中运行。同样在 url **[http://localhost:8501](http://localhost:8501)** 上来访问聊天机器人.

注意：不管你选择以上哪种安装方式，你都可以通过更改 `OLLAMA_API_URL` 环境变量（默认值为：http://localhost:11434）来指向你具体环境下的 ollama 服务地址。

快捷姿势：带有 hf_cache 的现成 [docker 镜像](https://hub.docker.com/repository/docker/faywong8888/deepseek-private-chatbot/)
---

# **这个聊天机器人如何工作**

1. **Upload Documents**: Add PDFs，DOCX，or TXT files via the sidebar。 
2. **Hybrid Retrieval**: Combines **BM25** and **FAISS** to fetch the most relevant text chunks。 
3. **GraphRAG Processing**: Builds a **Knowledge Graph** from your documents to understand relationships and context。
4. **Web Search**: Do a web search with [searxng](https://github.com/searxng/searxng)，and append it the context as data Source for refer。 
5. **Neural Reranking**: Uses a **Cross-Encoder** model for reordering the retrieved chunks by relevance。 
6. **Query Expansion (HyDE)**: Generates hypothetical answers to **expand** your query for better recall。 
7. **Chat Memory History Integration**: Maintains context by referencing previous user messages。 
8. **DeepSeek-8B Generation**: Produces the final answer based on top-ranked chunks.

---

## 📌 贡献 

- **克隆** 这个仓库，提交 **pull requests**，或创建 **issues**。 
---

### 🔗 分享想法

有反馈或建议，可以通过 [**V2EX**](https://www.v2ex.com/member/faywong8888) 联系我! 🚀💡

---