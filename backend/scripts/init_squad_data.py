"""
Initialize SQuAD v1.1 dev data as RAGInsight knowledge base.
Replaces the 20 synthetic documents with real Wikipedia paragraphs from SQuAD.

Prerequisites:
    data/squad_dev_v1.1.json  (downloaded from https://rajpurkar.github.io/SQuAD-explorer/)

Usage:
    cd backend && .\venv\Scripts\python -m scripts.init_squad_data
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import get_settings
from app.services.retriever import _get_embedding_fn

settings = get_settings()


def load_squad_documents(squad_path: str = "../data/squad_dev_v1.1.json"):
    with open(squad_path, "r", encoding="utf-8") as f:
        squad_data = json.load(f)

    documents = []
    for article in squad_data["data"]:
        title = article["title"].replace("_", " ")
        for para in article["paragraphs"]:
            context = para["context"].strip()
            if len(context) > 100:
                documents.append({
                    "content": context,
                    "source": f"SQuAD-{title}",
                })

    return documents


def main():
    squad_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "data", "squad_dev_v1.1.json"
    )

    if not os.path.exists(squad_path):
        print(f"ERROR: SQuAD file not found at {squad_path}")
        print("Please download it first:")
        print("  https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json")
        sys.exit(1)

    print("Loading SQuAD dev documents...")
    docs = load_squad_documents(squad_path)
    print(f"Loaded {len(docs)} paragraphs from SQuAD dev set")

    print("Indexing into ChromaDB (this may take a minute)...")
    client = chromadb.PersistentClient(
        path=settings.chroma_db_path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    embedding_fn = _get_embedding_fn()

    # Clear existing collection and rebuild
    try:
        client.delete_collection("nq_documents")
    except Exception:
        pass
    collection = client.get_or_create_collection(
        "nq_documents",
        embedding_function=embedding_fn,
    )

    documents_text = [d["content"] for d in docs]
    metadatas = [{"source": d["source"]} for d in docs]
    ids = [f"squad_{i}" for i in range(len(docs))]

    # Batch add to avoid memory issues
    batch_size = 100
    for i in range(0, len(documents_text), batch_size):
        end = min(i + batch_size, len(documents_text))
        batch_docs = documents_text[i:end]
        embeddings = embedding_fn(batch_docs)
        collection.add(
            documents=batch_docs,
            metadatas=metadatas[i:end],
            ids=ids[i:end],
            embeddings=embeddings,
        )
        print(f"  Indexed {end}/{len(documents_text)} documents...")

    print(f"\nDone! {len(docs)} real Wikipedia paragraphs now in ChromaDB.")
    print("Sample sources:")
    for d in docs[:5]:
        print(f"  - {d['source']}: {d['content'][:60]}...")


if __name__ == "__main__":
    main()
