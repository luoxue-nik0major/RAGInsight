import json
import os
import random
from zhconv import convert

random.seed(42)

DATA_DIR = r"d:\普通下载\大创\接下来主要的任务\data\chinese-poetry"
TARGET_COUNT = 1000

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def to_simplified(text):
    """Convert traditional Chinese to simplified."""
    return convert(text, 'zh-cn')

def format_poem(poem, dynasty):
    """Format a poem into a single text string (simplified Chinese)."""
    author = to_simplified(poem.get('author', '佚名'))
    title = to_simplified(poem.get('title', poem.get('rhythmic', '无题')))
    paragraphs = [to_simplified(p) for p in poem.get('paragraphs', [])]
    
    lines = [f"【{dynasty}】《{title}》 {author}"]
    for p in paragraphs:
        lines.append(p)
    return '\n'.join(lines)

poems = []

# 1. Load 唐诗三百首
tang300 = load_json(os.path.join(DATA_DIR, '全唐诗', '唐诗三百首.json'))
for p in tang300:
    poems.append({
        'text': format_poem(p, '唐诗'),
        'source': f"唐诗三百首 | {to_simplified(p.get('author','佚名'))}《{to_simplified(p.get('title','无题'))}》",
    })

# 2. Load 宋词三百首
song300 = load_json(os.path.join(DATA_DIR, '宋词', '宋词三百首.json'))
for p in song300:
    poems.append({
        'text': format_poem(p, '宋词'),
        'source': f"宋词三百首 | {to_simplified(p.get('author','佚名'))}《{to_simplified(p.get('rhythmic','无题'))}》",
    })

# 3. Supplement with more Tang poems from poet.tang.*.json
# Pick files that contain famous poets (skip poet.tang.0 which is mostly emperors)
tang_files = [
    'poet.tang.1000.json',
    'poet.tang.2000.json',
    'poet.tang.3000.json',
    'poet.tang.4000.json',
]

extra_tang = []
for fname in tang_files:
    path = os.path.join(DATA_DIR, '全唐诗', fname)
    if os.path.exists(path):
        data = load_json(path)
        extra_tang.extend(data)

# Shuffle and select enough to reach TARGET_COUNT
random.shuffle(extra_tang)
needed = TARGET_COUNT - len(poems)
for p in extra_tang[:needed]:
    poems.append({
        'text': format_poem(p, '唐诗'),
        'source': f"全唐诗 | {to_simplified(p.get('author','佚名'))}《{to_simplified(p.get('title','无题'))}》",
    })

# 4. If still not enough, add some Song poems
if len(poems) < TARGET_COUNT:
    song_files = sorted([
        f for f in os.listdir(os.path.join(DATA_DIR, '宋词'))
        if f.startswith('ci.song.') and f.endswith('.json') and f != '宋词三百首.json'
    ])
    extra_song = []
    for fname in song_files[:5]:  # first 5 files
        path = os.path.join(DATA_DIR, '宋词', fname)
        data = load_json(path)
        extra_song.extend(data)
    random.shuffle(extra_song)
    needed = TARGET_COUNT - len(poems)
    for p in extra_song[:needed]:
        poems.append({
            'text': format_poem(p, '宋词'),
            'source': f"全宋词 | {to_simplified(p.get('author','佚名'))}《{to_simplified(p.get('rhythmic','无题'))}》",
        })

print(f"Total poems collected: {len(poems)}")

# Generate init_data.py
output_path = r"d:\普通下载\大创\接下来主要的任务\backend\scripts\init_data.py"

lines = [
    '"""Initialize ChromaDB with Chinese classical poetry dataset."""',
    'import sys',
    'import os',
    '',
    '# Ensure project root is in path',
    'sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))',
    '',
    'from app.services.retriever import VectorRetrieverAdapter',
    '',
    '# Chinese classical poetry dataset (~1000 poems)',
    '# Sources: 唐诗三百首 + 宋词三百首 + supplemental poems from 全唐诗/全宋词',
    'SAMPLE_DOCUMENTS = [',
]

for p in poems:
    text = p['text'].replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    source = p['source'].replace('\\', '\\\\').replace('"', '\\"')
    lines.append(f'    {{')
    lines.append(f'        "content": "{text}",')
    lines.append(f'        "source": "{source}",')
    lines.append(f'    }},')

lines.extend([
    ']',
    '',
    'def main():',
    '    adapter = VectorRetrieverAdapter()',
    '    documents = [doc["content"] for doc in SAMPLE_DOCUMENTS]',
    '    metadatas = [{"source": doc["source"]} for doc in SAMPLE_DOCUMENTS]',
    '    ids = [f"poem_{i}" for i in range(len(SAMPLE_DOCUMENTS))]',
    '    print(f"Indexing {len(documents)} poems into ChromaDB...")',
    '    adapter.add_documents(documents, metadatas, ids)',
    '    print("Done! Poetry dataset indexed successfully.")',
    '',
    'if __name__ == "__main__":',
    '    main()',
])

with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Generated {output_path} with {len(poems)} poems (simplified Chinese).")
