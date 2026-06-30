"""
Batch import chinese-poetry dataset into ChromaDB.

Supports断点续传 (resume from checkpoint).
Expected runtime: ~10 minutes for 360k poems with batch=100.

Usage:
    cd backend
    python scripts/import_chinese_poetry.py
"""
import json
import os
import sys
import time
import glob
from typing import List, Dict, Any

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Must set BEFORE importing sentence-transformers
os.environ["HF_HOME"] = r"D:\1_A_work_code\embeddings"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import chromadb
from chromadb.config import Settings as ChromaSettings
from app.services.retriever import _get_embedding_fn

# ─── Configuration ──────────────────────────────────────────────────────────

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chinese-poetry")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "chinese_poetry"
BATCH_SIZE = 50
CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), "..", ".import_checkpoint.json")

# File patterns to import: (glob_pattern, dynasty_label, parser_type)
# NOTE: Limited to 唐诗(前5文件~5k首) + 宋词(~23k首) = ~28k total.
# Fits comfortably in systems with 4GB+ RAM.
FILE_PATTERNS = [
    # 全唐诗目录：poet.tang.* = 唐诗（只取前5个文件，约5000首）
    ("全唐诗/poet.tang.0.json", "唐诗", "poet"),
    ("全唐诗/poet.tang.1000.json", "唐诗", "poet"),
    ("全唐诗/poet.tang.10000.json", "唐诗", "poet"),
    ("全唐诗/poet.tang.11000.json", "唐诗", "poet"),
    ("全唐诗/poet.tang.12000.json", "唐诗", "poet"),
    # 宋词（全部23个文件）
    ("宋词/ci.song.*.json", "宋词", "ci"),
]

# ─── Parsers ─────────────────────────────────────────────────────────────────

def parse_poet(item: Dict) -> Dict[str, str]:
    """Parse poet.tang / poet.song format."""
    paragraphs = item.get("paragraphs", [])
    content = "\n".join(paragraphs)
    return {
        "title": item.get("title", "无题"),
        "author": item.get("author", "佚名"),
        "content": content,
    }

def parse_ci(item: Dict) -> Dict[str, str]:
    """Parse ci.song (宋词) format."""
    paragraphs = item.get("paragraphs", [])
    content = "\n".join(paragraphs)
    title = item.get("rhythmic", "无题")
    # Some ci entries also have a title override
    if "title" in item and item["title"]:
        title = f"{title}·{item['title']}"
    return {
        "title": title,
        "author": item.get("author", "佚名"),
        "content": content,
    }

def parse_qu(item: Dict) -> Dict[str, str]:
    """Parse 元曲 format."""
    paragraphs = item.get("paragraphs", [])
    content = "\n".join(paragraphs)
    return {
        "title": item.get("title", "无题"),
        "author": item.get("author", "佚名"),
        "content": content,
    }

def parse_shijing(item: Dict) -> Dict[str, str]:
    """Parse 诗经 format."""
    content_lines = item.get("content", [])
    content = "\n".join(content_lines)
    title = item.get("title", "无题")
    chapter = item.get("chapter", "")
    section = item.get("section", "")
    if chapter and section:
        title = f"{chapter}·{section}·{title}"
    return {
        "title": title,
        "author": "佚名",  # 诗经大多无明确作者
        "content": content,
    }

def parse_chuci(item: Dict) -> Dict[str, str]:
    """Parse 楚辞 format."""
    content_lines = item.get("content", [])
    content = "\n".join(content_lines)
    return {
        "title": item.get("title", "无题"),
        "author": item.get("author", "佚名"),
        "content": content,
    }

def parse_nalan(item: Dict) -> Dict[str, str]:
    """Parse 纳兰性德 format."""
    paragraphs = item.get("para", [])
    content = "\n".join(paragraphs)
    return {
        "title": item.get("title", "无题"),
        "author": item.get("author", "纳兰性德"),
        "content": content,
    }

def parse_yumengying(item: Dict) -> Dict[str, str]:
    """Parse 幽梦影 format (similar to poet)."""
    paragraphs = item.get("paragraphs", [])
    content = "\n".join(paragraphs)
    return {
        "title": item.get("title", "无题"),
        "author": item.get("author", "张潮"),
        "content": content,
    }

PARSERS = {
    "poet": parse_poet,
    "ci": parse_ci,
    "qu": parse_qu,
    "shijing": parse_shijing,
    "chuci": parse_chuci,
    "nalan": parse_nalan,
    "yumengying": parse_yumengying,
}

# ─── Core Logic ──────────────────────────────────────────────────────────────

def load_checkpoint() -> Dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_files": [], "total_imported": 0}

def save_checkpoint(cp: Dict):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)

def build_document(parsed: Dict[str, str], dynasty: str) -> tuple:
    """Build (document_text, metadata) from parsed poem."""
    title = parsed["title"]
    author = parsed["author"]
    content = parsed["content"]
    doc_text = f"【{dynasty}】《{title}》 {author}\n{content}"
    metadata = {
        "author": author,
        "dynasty": dynasty,
        "title": title,
        "source": f"chinese-poetry/{dynasty}",
    }
    return doc_text, metadata

def discover_files() -> List[tuple]:
    """Discover all JSON files to import."""
    files = []
    for pattern, dynasty, parser_type in FILE_PATTERNS:
        full_pattern = os.path.join(DATA_ROOT, pattern)
        matched = sorted(glob.glob(full_pattern))
        for filepath in matched:
            files.append((filepath, dynasty, parser_type))
    return files

def import_batch(collection, embedding_fn, docs: List[str], metas: List[Dict], ids: List[str]):
    """Import a batch of documents into ChromaDB."""
    import gc
    embeddings = embedding_fn(docs)
    collection.add(
        documents=docs,
        metadatas=metas,
        ids=ids,
        embeddings=embeddings,
    )
    gc.collect()  # Force garbage collection after each batch

def main():
    print("=" * 60)
    print("Chinese Poetry Batch Import Tool")
    print("=" * 60)

    # 1. Discover files
    all_files = discover_files()
    print(f"\n📁 Discovered {len(all_files)} JSON files to import")

    # 2. Load checkpoint
    cp = load_checkpoint()
    completed = set(cp.get("completed_files", []))
    total_imported = cp.get("total_imported", 0)
    print(f"📌 Checkpoint: {len(completed)} files already imported, {total_imported} poems total")

    # 3. Initialize ChromaDB
    print(f"\n🔌 Connecting to ChromaDB at {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection if requested via env
    if os.environ.get("RESET_COLLECTION", "") == "1":
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"🗑️  Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass
        completed = set()
        total_imported = 0

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Chinese classical poetry dataset"},
    )
    print(f"✅ Collection '{COLLECTION_NAME}' ready")

    # 4. Load embedding model
    print("\n🧠 Loading embedding model (BAAI/bge-small-zh-v1.5)...")
    embedding_fn = _get_embedding_fn()
    print("✅ Model loaded")

    # 5. Process files
    pending_files = [f for f in all_files if f[0] not in completed]
    print(f"\n🚀 Starting import: {len(pending_files)} files pending")
    print(f"   Batch size: {BATCH_SIZE}")
    print("-" * 60)

    file_idx = 0
    batch_docs = []
    batch_metas = []
    batch_ids = []

    for filepath, dynasty, parser_type in pending_files:
        file_idx += 1
        parser = PARSERS[parser_type]
        filename = os.path.basename(filepath)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"   ⚠️  SKIP {filename}: {e}")
            continue

        if not isinstance(data, list):
            print(f"   ⚠️  SKIP {filename}: not a list")
            continue

        poem_count = 0
        for item in data:
            try:
                parsed = parser(item)
                if not parsed["content"].strip():
                    continue
                doc_text, metadata = build_document(parsed, dynasty)
                poem_id = f"{dynasty}_{file_idx}_{poem_count}_{int(time.time()*1000)}"

                batch_docs.append(doc_text)
                batch_metas.append(metadata)
                batch_ids.append(poem_id)
                poem_count += 1

                if len(batch_docs) >= BATCH_SIZE:
                    import_batch(collection, embedding_fn, batch_docs, batch_metas, batch_ids)
                    total_imported += len(batch_docs)
                    batch_docs = []
                    batch_metas = []
                    batch_ids = []
            except Exception as e:
                # Skip malformed items
                continue

        # Mark file as completed
        completed.add(filepath)
        cp["completed_files"] = list(completed)
        cp["total_imported"] = total_imported
        save_checkpoint(cp)

        print(f"   [{file_idx:4d}/{len(pending_files):4d}] {dynasty:4s} | {filename:30s} | +{poem_count:5d} poems | total={total_imported:6d}")

    # 6. Flush remaining batch
    if batch_docs:
        import_batch(collection, embedding_fn, batch_docs, batch_metas, batch_ids)
        total_imported += len(batch_docs)
        cp["total_imported"] = total_imported
        save_checkpoint(cp)

    # 7. Final stats
    final_count = collection.count()
    print("\n" + "=" * 60)
    print("✅ IMPORT COMPLETE")
    print(f"   Total poems imported: {total_imported}")
    print(f"   Collection count: {final_count}")
    print(f"   Checkpoint file: {CHECKPOINT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
