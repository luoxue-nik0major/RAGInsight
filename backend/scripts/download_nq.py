"""
Download Natural Questions simplified dataset for RAGInsight knowledge base.

NOTE: This script requires internet access to HuggingFace (huggingface.co) 
or Google Cloud Storage. Run it in an environment with proper network access.

Prerequisites:
    pip install datasets

Usage:
    cd backend && .\venv\Scripts\python -m scripts.download_nq
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_nq_simplified(output_dir: str = "../data/nq_documents", max_examples: int = 500):
    """Download NQ simplified dataset and extract document chunks."""
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Natural Questions (simplified, streaming)...")
    ds = load_dataset(
        "google-research-datasets/natural_questions",
        "default",
        split="train",
        streaming=True,
    )

    documents = []
    questions = []

    for i, example in enumerate(ds):
        if i >= max_examples:
            break

        doc_text = example.get("document_text", "")
        question = example.get("question_text", "")
        doc_url = example.get("document_url", "")

        if not doc_text or len(doc_text) < 100:
            continue

        # Extract title from URL
        title = doc_url.split("/")[-1].replace("_", " ") if doc_url else f"NQ-Doc-{i}"

        # Split into paragraphs (long answer candidates)
        paragraphs = [
            p.strip()
            for p in doc_text.split("\n\n")
            if len(p.strip()) > 80 and len(p.strip()) < 2000
        ]

        for j, para in enumerate(paragraphs[:2]):  # Max 2 paragraphs per doc
            documents.append({
                "content": para,
                "source": f"NQ-{title}",
                "doc_index": i,
                "para_index": j,
            })

        questions.append({
            "question": question,
            "doc_title": title,
        })

    # Save documents
    doc_path = os.path.join(output_dir, "nq_documents.jsonl")
    with open(doc_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Save questions
    q_path = os.path.join(output_dir, "nq_questions.jsonl")
    with open(q_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nDone!")
    print(f"  Documents: {len(documents)} chunks -> {doc_path}")
    print(f"  Questions: {len(questions)} questions -> {q_path}")
    print(f"\nNext step: Run 'python -m scripts.init_data' to index these into ChromaDB.")


if __name__ == "__main__":
    download_nq_simplified()
