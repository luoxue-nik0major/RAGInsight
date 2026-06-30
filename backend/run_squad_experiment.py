"""
Run Phase 5 experiments with SQuAD 200 queries.
"""
import asyncio
import json
import requests
import time
import sys

BASE = "http://localhost:8000"

def check_backend():
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception as e:
        print(f"Backend not reachable: {e}")
        return False

def load_dataset():
    r = requests.get(f"{BASE}/api/experiments/dataset")
    data = r.json()
    return data["queries"], data["count"]

def start_experiment():
    r = requests.post(f"{BASE}/api/experiments/run")
    return r.json()

def get_status():
    r = requests.get(f"{BASE}/api/experiments/status")
    return r.json()

def get_results():
    r = requests.get(f"{BASE}/api/experiments/results")
    return r.json()

def get_metrics():
    r = requests.get(f"{BASE}/api/experiments/metrics")
    return r.json()

def main():
    print("=" * 60)
    print("SQuAD Phase 5 Experiment Runner")
    print("=" * 60)
    
    if not check_backend():
        print("ERROR: Backend not running on port 8000")
        print("Please start: cd backend && .\\venv\\Scripts\\python -m uvicorn app.main:app --port 8000")
        sys.exit(1)
    
    print("\n[1] Loading dataset...")
    queries, count = load_dataset()
    print(f"    Loaded {count} queries from SQuAD dataset")
    print(f"    Sample: [{queries[0]['id']}] {queries[0]['query'][:50]}...")
    
    # Check if experiment already running
    status = get_status()
    if status.get("is_running"):
        print("\n[!] Experiment already running. Monitoring...")
    else:
        print("\n[2] Starting experiment batch (200 queries x 2 strategies = 400 runs)...")
        result = start_experiment()
        print(f"    Status: {result.get('status')}")
        print(f"    Total: {result.get('total_queries')} queries")
        print(f"    Strategies: {result.get('strategies')}")
    
    print("\n[3] Monitoring progress (Ctrl+C to stop monitoring)...")
    last_current = -1
    stable_count = 0
    while True:
        status = get_status()
        current = status.get("current", 0)
        total = status.get("total", 0)
        st = status.get("status", "unknown")
        
        if current != last_current:
            pct = (current / total * 100) if total else 0
            print(f"    [{st}] {current}/{total} ({pct:.1f}%)")
            last_current = current
            stable_count = 0
        else:
            stable_count += 1
        
        if st == "completed":
            print(f"\n[4] Experiment COMPLETED! {current}/{total} runs finished.")
            break
        if not status.get("is_running") and st != "running":
            print(f"\n[4] Experiment ended (status={st}).")
            break
        
        time.sleep(5)
    
    print("\n[5] Fetching results...")
    results = get_results()
    print(f"    Results in memory: {results['count']}")
    
    print("\n[6] Computing metrics...")
    metrics = get_metrics()
    
    fd = metrics.get("experiment_1_fault_detection")
    sc = metrics.get("experiment_2_strategy_comparison")
    
    if fd:
        print("\n--- Experiment 1: Fault Detection ---")
        macro = fd.get("macro", {})
        micro = fd.get("micro", {})
        print(f"    Macro Precision: {macro.get('precision', 0):.4f}")
        print(f"    Macro Recall:    {macro.get('recall', 0):.4f}")
        print(f"    Macro F1:        {macro.get('f1', 0):.4f}")
        print(f"    Micro Precision: {micro.get('precision', 0):.4f}")
        print(f"    Micro Recall:    {micro.get('recall', 0):.4f}")
        print(f"    Micro F1:        {micro.get('f1', 0):.4f}")
        print(f"    Per-alert breakdown:")
        for alert_type, stats in fd.get("per_alert_type", {}).items():
            print(f"      {alert_type:20s} P={stats['precision']:.4f} R={stats['recall']:.4f} F1={stats['f1']:.4f} (tp={stats['tp']} fp={stats['fp']} fn={stats['fn']})")
    
    if sc:
        print("\n--- Experiment 2: Strategy Comparison ---")
        for strategy, stats in sc.get("strategies", {}).items():
            print(f"    Strategy: {strategy}")
            print(f"      Runs:           {stats['count']}")
            print(f"      Avg Relevance:  {stats['avg_relevance']:.4f}")
            print(f"      Avg Coverage:   {stats['avg_coverage']:.4f}")
            print(f"      Avg Diversity:  {stats['avg_diversity']:.4f}")
            print(f"      Avg Quality:    {stats['avg_answer_quality']:.4f}")
            print(f"      Avg Duration:   {stats['avg_duration_ms']:.1f}ms")
    
    print("\n[7] Experiment file saved to backend/experiments/")
    print("=" * 60)
    print("DONE")

if __name__ == "__main__":
    main()
