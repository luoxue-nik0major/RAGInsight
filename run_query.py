#!/usr/bin/env python3
"""
RAGInsight 查询执行脚本
配置下方 QUERY_CONFIG 后，直接运行即可执行一次检索

用法:
    python run_query.py

依赖: 仅需 Python 标准库 (urllib, json)
"""

import json
import urllib.request
import urllib.error

# ==================== 查询配置 ====================
# 修改这里的参数来执行不同的查询

QUERY_CONFIG = {
    # 查询文本（必填）
    "query": "李白的诗歌风格是怎样的",
    
    # 检索策略: "vector" | "hybrid" | "graph"
    "strategy": "vector",
    
    # 指定集合（可选，留空使用默认）
    "collection": None,
    
    # 后端地址
    "base_url": "http://localhost:8000",
}

# ==================== 查询执行 ====================


def print_header(title: str):
    """打印带分隔线的标题"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_section(title: str):
    """打印小节标题"""
    print(f"\n  ▶ {title}")
    print(f"  {'─' * 50}")


def run_query(config: dict):
    """执行 RAG 查询并解析 SSE 响应"""
    base_url = config["base_url"].rstrip("/")
    url = f"{base_url}/api/query"
    
    payload = json.dumps({
        "query": config["query"],
        "strategy": config["strategy"],
        "collection": config.get("collection"),
    }).encode("utf-8")
    
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    
    print_header("RAGInsight 查询执行")
    print(f"  查询: {config['query']}")
    print(f"  策略: {config['strategy']}")
    print(f"  API:  {url}")
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            # 解析 SSE 流
            steps = []
            alerts = []
            root_cause = None
            done_data = None
            complexity = None
            recommendation = None
            
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                
                data_str = line[6:]  # 去掉 "data: " 前缀
                if data_str == "[DONE]":
                    break
                
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                
                event_type = event.get("event")
                event_data = event.get("data", {})
                
                if event_type == "session_created":
                    complexity = event_data.get("complexity", {})
                    recommendation = event_data.get("recommendation", {})
                
                elif event_type == "step":
                    steps.append(event_data)
                
                elif event_type == "alert":
                    alerts.append(event_data)
                
                elif event_type == "root_cause":
                    root_cause = event_data
                
                elif event_type == "done":
                    done_data = event_data
            
            # ============ 输出结果 ============
            
            # 1. 复杂度分析与策略推荐
            if complexity:
                print_section("查询复杂度分析")
                score = complexity.get("complexity_score", 0)
                qtype = complexity.get("question_type", "unknown")
                print(f"  复杂度分数: {score:.1%}")
                print(f"  问题类型:   {qtype}")
                features = complexity.get("features", {})
                for key, val in features.items():
                    if key.endswith("_score") and isinstance(val, (int, float)):
                        print(f"  {key:12s}: {val:.2f}")
            
            if recommendation:
                print(f"\n  推荐策略: {recommendation.get('recommended_strategy_name', '未知')}")
                print(f"  推荐理由: {recommendation.get('reason', '无')}")
            
            # 2. 流程步骤
            if steps:
                print_section("检索流程步骤")
                for step in steps:
                    stype = step.get("step_type", "unknown")
                    duration = step.get("duration_ms", 0)
                    qscore = step.get("quality_score")
                    qinfo = f"Q: {qscore:.0%}" if qscore is not None else "Q: N/A"
                    print(f"  • {stype:20s} {duration:5d}ms  {qinfo}")
                    
                    # 打印检索到的 chunks
                    chunks = step.get("chunks", [])
                    if chunks:
                        for i, chunk in enumerate(chunks):
                            content = chunk.get("content", "")[:60].replace("\n", " ")
                            rel = chunk.get("relevance_score", 0)
                            src = chunk.get("source", "")
                            print(f"    Chunk {i}: [Rel:{rel:.0%}] {content}...")
                            if src:
                                print(f"             来源: {src}")
            
            # 3. 警告
            if alerts:
                print_section(f"质量警告 ({len(alerts)} 个)")
                for alert in alerts:
                    severity = alert.get("severity", "warning")
                    icon = "⚠️" if severity == "warning" else "❌"
                    print(f"  {icon} [{alert.get('alert_type', 'unknown')}]")
                    print(f"     消息: {alert.get('message', '')}")
                    suggestion = alert.get("suggestion")
                    if suggestion:
                        print(f"     建议: {suggestion}")
            else:
                print_section("质量警告")
                print("  ✅ 无警告")
            
            # 4. 根因分析
            if root_cause:
                print_section("根因分析")
                print(f"  类型: {root_cause.get('root_cause_label', '未知')}")
                print(f"  严重级别: {root_cause.get('severity', 'unknown')}")
                print(f"  说明: {root_cause.get('explanation', '')}")
                suggestions = root_cause.get("suggestions", [])
                if suggestions:
                    print(f"  修复建议:")
                    for s in suggestions:
                        print(f"    • {s}")
            
            # 5. 最终答案
            if done_data:
                print_section("最终答案")
                answer = done_data.get("answer", "")
                print(f"  {answer}")
                
                # 6. 答案评估
                eval_info = done_data.get("answer_evaluation", {})
                if eval_info:
                    print_section("答案质量评估")
                    faith = eval_info.get("faithfulness", {})
                    rel = eval_info.get("relevance", {})
                    combined = eval_info.get("combined_score", 0)
                    print(f"  忠实度:   {faith.get('score', 0):.0%} "
                          f"({faith.get('supported_claims', 0)}/{faith.get('total_claims', 0)} claims supported)")
                    print(f"  相关性:   {rel.get('score', 0):.0%} ({rel.get('method', '')})")
                    print(f"  综合评分: {combined:.0%}")
            
            print_header("查询完成")
    
    except urllib.error.URLError as e:
        print(f"\n[错误] 无法连接到后端: {e}")
        print(f"请确保后端已启动: python start_project.py")
        print(f"或检查 base_url 配置: {config['base_url']}")
    except Exception as e:
        print(f"\n[错误] 查询执行失败: {e}")


def main():
    run_query(QUERY_CONFIG)


if __name__ == "__main__":
    main()
