import re
from typing import List


def is_chinese_text(text: str) -> bool:
    """Check if text is predominantly Chinese (>30% Chinese characters)."""
    if not text:
        return False
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    return chinese_chars / len(text) > 0.3


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text for BM25. Supports Chinese (jieba) and English (whitespace)."""
    if is_chinese_text(text):
        import jieba
        return list(jieba.cut_for_search(text))
    else:
        # English: lowercase, extract words with 2+ chars
        return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
