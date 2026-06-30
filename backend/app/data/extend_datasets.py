"""Extend test datasets with diverse query types."""
import json
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# === Extend English SQuAD dataset ===
with open(os.path.join(DATA_DIR, "squad_test_queries.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

existing_ids = {q["id"] for q in data["queries"]}

new_english = [
    # Multi-hop queries
    {"id": 201, "query": "What team did Von Miller play for, and who was their opponent in Super Bowl 50?", "category": "multi_hop", "expected_answer_type": "relational", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": []},
    {"id": 202, "query": "Which stadium hosted Super Bowl 50, and what year did it open?", "category": "multi_hop", "expected_answer_type": "relational", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": []},
    {"id": 203, "query": "Who was the MVP of Super Bowl 50 and which team did he play for?", "category": "multi_hop", "expected_answer_type": "relational", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": []},
    {"id": 204, "query": "Which network broadcast Super Bowl 50 and who headlined the halftime show?", "category": "multi_hop", "expected_answer_type": "relational", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": []},
    {"id": 205, "query": "What was the cost of a 30-second ad during Super Bowl 50, and which network aired it?", "category": "multi_hop", "expected_answer_type": "relational", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": []},
    # Comparison queries
    {"id": 206, "query": "Compare the regular season records of the Panthers and the Broncos leading up to Super Bowl 50.", "category": "comparison", "expected_answer_type": "comparative", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": []},
    {"id": 207, "query": "What are the differences between the AFC and NFC champions paths to Super Bowl 50?", "category": "comparison", "expected_answer_type": "comparative", "expected_depth": 2, "recommended_strategy": "graph", "expected_alerts": ["low_relevance"]},
    {"id": 208, "query": "Compare the three candidate stadiums for hosting Super Bowl 50.", "category": "comparison", "expected_answer_type": "comparative", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": []},
    {"id": 209, "query": "How do the halftime performers at Super Bowl 50 compare to each other?", "category": "comparison", "expected_answer_type": "comparative", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": ["low_relevance"]},
    {"id": 210, "query": "Compare the Denver Broncos performance in Super Bowl XLVIII vs Super Bowl 50.", "category": "comparison", "expected_answer_type": "comparative", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": []},
    # Unanswerable queries
    {"id": 211, "query": "What was the name of the halftime show producer for Super Bowl 50?", "category": "unanswerable", "expected_answer_type": "none", "expected_depth": 0, "recommended_strategy": "vector", "expected_alerts": ["empty_results", "low_relevance"]},
    {"id": 212, "query": "Who sang the national anthem at Super Bowl 50?", "category": "unanswerable", "expected_answer_type": "none", "expected_depth": 0, "recommended_strategy": "vector", "expected_alerts": ["empty_results"]},
    {"id": 213, "query": "What was the attendance figure for Super Bowl 50?", "category": "unanswerable", "expected_answer_type": "none", "expected_depth": 0, "recommended_strategy": "vector", "expected_alerts": ["empty_results"]},
    {"id": 214, "query": "How many hot dogs were sold at Super Bowl 50?", "category": "unanswerable", "expected_answer_type": "none", "expected_depth": 0, "recommended_strategy": "vector", "expected_alerts": ["empty_results", "low_relevance"]},
    {"id": 215, "query": "What was the weather temperature during Super Bowl 50?", "category": "unanswerable", "expected_answer_type": "none", "expected_depth": 0, "recommended_strategy": "vector", "expected_alerts": ["empty_results"]},
    # Edge-case queries
    {"id": 216, "query": "Super Bowl 50 Super Bowl 50 Super Bowl 50?", "category": "unanswerable", "expected_answer_type": "none", "expected_depth": 0, "recommended_strategy": "vector", "expected_alerts": ["low_relevance"]},
    {"id": 217, "query": "ablkjsdfkjlkjlkj sdfjlkjsdf?", "category": "unanswerable", "expected_answer_type": "none", "expected_depth": 0, "recommended_strategy": "vector", "expected_alerts": ["empty_results", "low_relevance"]},
    # Causal queries
    {"id": 218, "query": "Why did the Florida legislature refuse to fund the Sun Life Stadium renovation?", "category": "causal", "expected_answer_type": "explanatory", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": ["low_relevance"]},
    {"id": 219, "query": "What factors led to the Denver Broncos winning Super Bowl 50?", "category": "causal", "expected_answer_type": "explanatory", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": []},
    {"id": 220, "query": "Why was Super Bowl 50 called the golden anniversary?", "category": "causal", "expected_answer_type": "explanatory", "expected_depth": 1, "recommended_strategy": "vector", "expected_alerts": []},
    # Long/complex queries
    {"id": 221, "query": "Can you give me a very detailed description of everything that happened during the Super Bowl 50 game including the key players, their statistics, the halftime show, the commercials, and the final outcome?", "category": "simple_fact", "expected_answer_type": "factual", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": []},
    {"id": 222, "query": "Tell me all the details about Super Bowl 50 including who played, where it was held, who won, who performed at halftime, and what network broadcast the event.", "category": "simple_fact", "expected_answer_type": "factual", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": []},
    # Definition queries
    {"id": 223, "query": "What is the Super Bowl?", "category": "simple_fact", "expected_answer_type": "definition", "expected_depth": 1, "recommended_strategy": "vector", "expected_alerts": []},
    {"id": 224, "query": "What does NFL stand for and what is its significance?", "category": "simple_fact", "expected_answer_type": "definition", "expected_depth": 1, "recommended_strategy": "vector", "expected_alerts": []},
    {"id": 225, "query": "Define what a linebacker does in American football.", "category": "unanswerable", "expected_answer_type": "definition", "expected_depth": 0, "recommended_strategy": "vector", "expected_alerts": ["empty_results", "low_relevance"]},
]

# Ensure no ID conflicts
for q in new_english:
    assert q["id"] not in existing_ids, f"Duplicate ID: {q['id']}"

data["queries"].extend(new_english)
data["description"] = f"Extended SQuAD test queries with {len(data['queries'])} entries covering simple_fact, multi_hop, comparison, causal, and unanswerable types."

with open(os.path.join(DATA_DIR, "squad_test_queries.json"), "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"English dataset: {len(data['queries'])} queries (was 200, added {len(new_english)})")

# === Extend Chinese dataset ===
with open(os.path.join(DATA_DIR, "chinese_test_queries.json"), "r", encoding="utf-8") as f:
    ch_data = json.load(f)

ch_existing_ids = {q["id"] for q in ch_data["queries"]}

new_chinese = [
    # Multi-hop
    {"id": 131, "query": "杜甫在安史之乱期间写了哪些诗，这些诗反映了什么社会现实？", "category": "multi_hop", "expected_answer_type": "relational", "expected_depth": 2, "recommended_strategy": "graph", "expected_alerts": ["low_relevance"]},
    {"id": 132, "query": "王之涣的《登鹳雀楼》与他的边塞诗在主题上有什么联系？", "category": "multi_hop", "expected_answer_type": "relational", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": ["low_relevance"]},
    {"id": 133, "query": "苏轼的《水调歌头》与他的贬谪经历有什么关系？", "category": "multi_hop", "expected_answer_type": "relational", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": []},
    {"id": 134, "query": "李白的借酒抒怀与传统文人诗酒文化有什么关系？", "category": "multi_hop", "expected_answer_type": "relational", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": []},
    {"id": 135, "query": "白居易的新乐府运动与他的政治主张有什么关联？", "category": "multi_hop", "expected_answer_type": "relational", "expected_depth": 2, "recommended_strategy": "graph", "expected_alerts": ["low_relevance"]},
    # Comparison
    {"id": 136, "query": "比较唐诗和宋词在意象运用上的差异", "category": "comparison", "expected_answer_type": "comparative", "expected_depth": 2, "recommended_strategy": "graph", "expected_alerts": ["low_relevance"]},
    {"id": 137, "query": "李白的绝句与杜甫的律诗在艺术手法上有什么不同？", "category": "comparison", "expected_answer_type": "comparative", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": []},
    # Unanswerable
    {"id": 138, "query": "唐代诗人有没有写过关于元宇宙的作品？", "category": "unanswerable", "expected_answer_type": "none", "expected_depth": 0, "recommended_strategy": "vector", "expected_alerts": ["empty_results"]},
    {"id": 139, "query": "李白写过关于相对论的诗吗？", "category": "unanswerable", "expected_answer_type": "none", "expected_depth": 0, "recommended_strategy": "vector", "expected_alerts": ["empty_results"]},
    # Long/edge-case
    {"id": 140, "query": "请详细描述唐诗宋词的发展脉络，包括初唐、盛唐、中唐、晚唐的代表诗人及其风格特点，以及宋词的豪放派和婉约派的主要区别。", "category": "simple_fact", "expected_answer_type": "factual", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": []},
    {"id": 141, "query": "唐代诗人是唐代诗人吗唐代诗人唐代？", "category": "unanswerable", "expected_answer_type": "none", "expected_depth": 0, "recommended_strategy": "vector", "expected_alerts": ["low_relevance"]},
    # Definition
    {"id": 142, "query": "什么是近体诗？", "category": "simple_fact", "expected_answer_type": "definition", "expected_depth": 1, "recommended_strategy": "vector", "expected_alerts": []},
    {"id": 143, "query": "什么是词的婉约派？", "category": "simple_fact", "expected_answer_type": "definition", "expected_depth": 1, "recommended_strategy": "vector", "expected_alerts": []},
    {"id": 144, "query": "诗歌中的意象是什么？", "category": "simple_fact", "expected_answer_type": "definition", "expected_depth": 1, "recommended_strategy": "vector", "expected_alerts": []},
    # Causal
    {"id": 145, "query": "为什么杜甫被称为诗史？", "category": "causal", "expected_answer_type": "explanatory", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": []},
    {"id": 146, "query": "为什么说李白的诗具有浪漫主义风格？", "category": "causal", "expected_answer_type": "explanatory", "expected_depth": 2, "recommended_strategy": "hybrid", "expected_alerts": []},
    # More simple facts
    {"id": 147, "query": "初唐四杰指的是哪些诗人？", "category": "simple_fact", "expected_answer_type": "factual", "expected_depth": 1, "recommended_strategy": "vector", "expected_alerts": []},
    {"id": 148, "query": "唐代边塞诗的代表诗人有哪些？", "category": "simple_fact", "expected_answer_type": "factual", "expected_depth": 1, "recommended_strategy": "vector", "expected_alerts": []},
    {"id": 149, "query": "元稹和白居易共同倡导了什么文学运动？", "category": "simple_fact", "expected_answer_type": "factual", "expected_depth": 1, "recommended_strategy": "vector", "expected_alerts": []},
    {"id": 150, "query": "宋词的豪放派代表人物有哪些？", "category": "simple_fact", "expected_answer_type": "factual", "expected_depth": 1, "recommended_strategy": "vector", "expected_alerts": []},
]

for q in new_chinese:
    assert q["id"] not in ch_existing_ids, f"Duplicate ID: {q['id']}"

ch_data["queries"].extend(new_chinese)
ch_data["description"] = f"Extended Chinese poetry test queries with {len(ch_data['queries'])} entries covering simple_fact, multi_hop, comparison, unanswerable, causal, and definition types."

with open(os.path.join(DATA_DIR, "chinese_test_queries.json"), "w", encoding="utf-8") as f:
    json.dump(ch_data, f, ensure_ascii=False, indent=2)
print(f"Chinese dataset: {len(ch_data['queries'])} queries (was 30, added {len(new_chinese)})")

print("\n=== Dataset Statistics ===")
for name, path in [("English", "squad_test_queries.json"), ("Chinese", "chinese_test_queries.json")]:
    with open(os.path.join(DATA_DIR, path), "r", encoding="utf-8") as f:
        d = json.load(f)
    cats = {}
    for q in d["queries"]:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    print(f"{name}: {len(d['queries'])} queries, categories: {cats}")
