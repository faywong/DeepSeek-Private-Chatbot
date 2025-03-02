import streamlit as st
import requests
import json
from utils.retriever_pipeline import retrieve_documents
from utils.doc_handler import process_documents
from sentence_transformers import CrossEncoder
import torch
import os
from dotenv import load_dotenv, find_dotenv
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]  # Fix for torch classes not found error
load_dotenv(find_dotenv())  # Loads .env file contents into the application based on key-value pairs defined therein, making them accessible via 'os' module functions like os.getenv().

OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
MODEL= os.getenv("MODEL", "deepseek-r1:7b")                                                      #Make sure you have it installed in ollama
EMBEDDINGS_MODEL = "rjmalagon/gte-qwen2-1.5b-instruct-embed-f16:latest"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-large"
SEARXNG_URL = os.getenv("SEARXNG_API_URL", "http://searxng:8080/search")

device = "cuda" if torch.cuda.is_available() else "cpu"

reranker = None                                                        # 🚀 Initialize Cross-Encoder (Reranker) at the global level 
# note: add local_files_only=True in docker build to use hf_cache
try:
    reranker = CrossEncoder(CROSS_ENCODER_MODEL, cache_dir="./hf_cache", device=device)
except Exception as e:
    st.error(f"Failed to load CrossEncoder model: {str(e)}")


st.set_page_config(page_title="DeepGraph RAG-Pro", layout="wide")      # ✅ Streamlit configuration

# Custom CSS
st.markdown("""
    <style>
        .stApp { background-color: #f4f4f9; }
        h1 { color: #00FF99; text-align: center; }
        .stChatMessage { border-radius: 10px; padding: 10px; margin: 10px 0; }
        .stChatMessage.user { background-color: #e8f0fe; }
        .stChatMessage.assistant { background-color: #d1e7dd; }
        .stButton>button { background-color: #00AAFF; color: white; }
    </style>
""", unsafe_allow_html=True)
                                                                                    # Manage Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_pipeline" not in st.session_state:
    st.session_state.retrieval_pipeline = None
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False
if "search_enabled" not in st.session_state:
    st.session_state.search_enabled = False
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

def format_search_results(results: list, max_results: int = 10) -> str:
    """
    Format the top search results into a context string.
    """
    formatted = []
    for result in results[:max_results]:
        title = result.get("title", "No title")
        url = result.get("url", "No URL")
        snippet = result.get("content", "No snippet")
        formatted.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}")
    return "\n\n".join(formatted)

def search_web(query: str) -> list:
    params = {'q': query, 'format': 'json'}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    response = requests.get(SEARXNG_URL, params=params, headers=headers)
    # print(f"response: {response.text}")
    if response.status_code != 200:
        print("Response status code:", response.status_code)
        print("Response text:", response.text)
        raise Exception(f"Search query failed with status code {response.status_code}")
    return response.json().get("results", [])

with st.sidebar:                                                                        # 📁 Sidebar
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files and not st.session_state.documents_loaded:
        with st.spinner("Processing documents..."):
            process_documents(uploaded_files,reranker,EMBEDDINGS_MODEL, OLLAMA_BASE_URL)
            st.success("Documents processed!")
    
    st.markdown("---")
    st.header("⚙️ RAG Settings")

    st.session_state.search_enabled = st.checkbox("Enable Search", value=True)
    st.session_state.rag_enabled = st.checkbox("Enable RAG", value=True)
    st.session_state.enable_hyde = st.checkbox("Enable HyDE", value=True)
    st.session_state.enable_reranking = st.checkbox("Enable Neural Reranking", value=True)
    st.session_state.enable_graph_rag = st.checkbox("Enable GraphRAG", value=True)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    st.session_state.max_contexts = st.slider("Max Contexts", 1, 5, 3)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # 🚀 Footer (Bottom Right in Sidebar) For some Credits :)
    st.sidebar.markdown("""
        <div style="position: absolute; top: 20px; right: 10px; font-size: 12px; color: gray;">
            <b>Developed by:</b> N Sai Akhil &copy; All Rights Reserved 2025
        </div>
    """, unsafe_allow_html=True)

# 💬 Chat Interface
st.title("🤖 DeepGraph RAG-Pro")
st.caption("Advanced RAG System with GraphRAG, Hybrid Retrieval, Neural Reranking and Chat History")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    chat_history = "\n".join([msg["content"] for msg in st.session_state.messages[-5:]])  # Last 5 messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        print(f"chat_message prompt:{prompt}, st.session_state.rag_enabled: {st.session_state.rag_enabled}, st.session_state.retrieval_pipeline:{st.session_state.retrieval_pipeline}, st.session_state.search_enabled: {st.session_state.search_enabled}")

        cur_source_idx = 1
        # 🚀 Build context
        context = ""
        if st.session_state.search_enabled:
            search_results = search_web(prompt)
            context = f"[Source {cur_source_idx}]: " + format_search_results(search_results, max_results=10)
        # print(f"ctx after web search: {context}")
        cur_source_idx += 1
        if st.session_state.rag_enabled and st.session_state.retrieval_pipeline:
            try:
                docs = retrieve_documents(prompt, OLLAMA_API_URL, MODEL, chat_history)
                context = context + "\n" + "\n".join(
                    f"[Source {cur_source_idx + i + i}]: {doc.page_content}" 
                    for i, doc in enumerate(docs)
                )
            except Exception as e:
                st.error(f"Retrieval error: {str(e)}")
        
        # 🚀 Structured Prompt
        system_prompt = f"""Use the chat history to maintain context:
            Chat History:
            {chat_history}

            Analyze the question and context through these steps:
            1. Identify key entities and relationships
            2. Check for contradictions between sources
            3. Synthesize information from multiple contexts
            4. Formulate a structured response

            Context:
            {context}

            Question: {prompt}
            Answer:"""
        
        # Stream response
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL,
                "prompt": system_prompt,
                "stream": True,
                "options": {
                    "temperature": st.session_state.temperature,  # Use dynamic user-selected value
                    "num_ctx": 4096
                }
            },
            stream=True
        )
        try:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode())
                    token = data.get("response", "")
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                    
                    # Stop if we detect the end token
                    if data.get("done", False):
                        break
                        
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Generation error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error."})
