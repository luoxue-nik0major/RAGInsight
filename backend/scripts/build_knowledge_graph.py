"""Build a lightweight knowledge graph from ChromaDB collection."""
import sys
import os
import json
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jieba.posseg as pseg
import networkx as nx
from app.services.retriever import VectorRetrieverAdapter


def extract_entities(text: str):
    """Extract Chinese entities using jieba POS tagging."""
    entities = []
    for word, flag in pseg.cut(text):
        if flag in ("nr", "ns", "nt", "nz"):  # person, place, org, proper noun
            entities.append(word)
    return list(set(entities))


def build_graph():
    adapter = VectorRetrieverAdapter()

    G = nx.Graph()

    # Build from both collections (English + Chinese)
    for collection in (adapter._collection_en, adapter._collection_zh):
        all_data = collection.get(include=["documents", "metadatas"])

        for idx, (doc_id, content, metadata) in enumerate(zip(all_data["ids"], all_data["documents"], all_data["metadatas"])):
            # Add chunk node
            chunk_node = f"chunk:{doc_id}"
            G.add_node(chunk_node, node_type="chunk", content=content[:200], source=metadata.get("source", ""))

            # Extract entities
            entities = extract_entities(content)
            for ent in entities:
                ent_node = f"ent:{ent}"
                if not G.has_node(ent_node):
                    G.add_node(ent_node, node_type="entity", name=ent)
                # Link entity to chunk
                G.add_edge(ent_node, chunk_node, edge_type="contains")

            # Link co-occurring entities
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1:]:
                    e1_node = f"ent:{e1}"
                    e2_node = f"ent:{e2}"
                    if G.has_edge(e1_node, e2_node):
                        G[e1_node][e2_node]["weight"] = G[e1_node][e2_node].get("weight", 0) + 1
                    else:
                        G.add_edge(e1_node, e2_node, edge_type="cooccurs", weight=1)

    return G


def serialize_graph(G: nx.Graph, output_path: str):
    """Serialize graph to JSON-friendly format."""
    data = {
        "nodes": [
            {"id": n, **G.nodes[n]}
            for n in G.nodes()
        ],
        "edges": [
            {"source": u, "target": v, **d}
            for u, v, d in G.edges(data=True)
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def main():
    output_path = os.path.join(os.path.dirname(__file__), "..", "app", "data", "knowledge_graph.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Building knowledge graph from ChromaDB collection...")
    G = build_graph()
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    serialize_graph(G, output_path)
    print(f"Graph saved to {output_path}")


if __name__ == "__main__":
    main()
