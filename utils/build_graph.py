import streamlit as st
import networkx as nx
import re

def build_knowledge_graph(docs):
    G = nx.Graph()
    for doc in docs:
        paragraphs = re.split(r'[\n,，。.]+', doc.page_content)
        entities = [p.strip() for p in paragraphs if p.strip()]
        # print(f"entities: {entities}")
        # Ensure meaningful relationships exist
        if len(entities) > 1:
            for i in range(len(entities) - 1):
                G.add_edge(entities[i], entities[i + 1])  # Create edge
    return G


def retrieve_from_graph(query, G, top_k=5):
    st.write(f"🔎 Searching GraphRAG for: {query}")

    # Convert query into words to match knowledge graph nodes
    query_words = query.lower().split()
    matched_nodes = [node for node in G.nodes if any(word in node.lower() for word in query_words)]
    
    if matched_nodes:
        related_nodes = []
        for node in matched_nodes:
            related_nodes.extend(list(G.neighbors(node)))  # Get connected nodes
        
        st.write(f"🟢 GraphRAG Matched Nodes: {matched_nodes}")
        st.write(f"🟢 GraphRAG Retrieved Related Nodes: {related_nodes[:top_k]}")
        return related_nodes[:top_k]
    
    st.write(f"❌ No graph results found for: {query}")
    return []
