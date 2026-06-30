"""Launch SQuAD experiment and monitor until completion."""
import requests, time, sys

BASE = "http://localhost:8000"

def main():
    # 1. Verify dataset
    r = requests.get(f"{BASE}/api/experiments/dataset")
    d = r.json()
    print(f"Dataset loaded: {d['count']} queries")
    print(f"First: {d['queries'][0]['query'][:60]}...")
    
    # 2. Check if already running
    status = requests.get(f"{BASE}/api/experiments/status").json()
    if status.get("is_running"):
        print(f"Already running: {status['current']}/{status['total']}")
    else:
        # 3. Start experiment
        print("Starting experiment batch...")
        result = requests.post(f"{BASE}/api/experiments/run").json()
        print(f"Started: {result.get('total_queries')} queries x {result.get('strategies')} strategies")
    
    # 4. Monitor
    print("\nMonitoring progress (updates every 10s):")
    last = -1
    while True:
        status = requests.get(f"{BASE}/api/experiments/status").json()
        curr = status.get("current", 0)
        total = status.get("total", 0)
        st = status.get("status", "unknown")
        
        if curr != last or st in ("completed", "idle"):
            pct = (curr/total*100) if total else 0
            print(f"  [{st}] {curr}/{total} ({pct:.1f}%)")
            last = curr
        
        if st == "completed" or (not status.get("is_running") and st != "running"):
            print("\nExperiment finished!")
            # Fetch metrics
            m = requests.get(f"{BASE}/api/experiments/metrics").json()
            print(f"Total runs: {m.get('total_runs')}")
            fd = m.get("experiment_1_fault_detection")
            if fd:
                macro = fd.get("macro", {})
                print(f"Macro F1: {macro.get('f1', 0):.4f}")
            sc = m.get("experiment_2_strategy_comparison")
            if sc:
                for strat, stats in sc.get("strategies", {}).items():
                    print(f"{strat}: relevance={stats['avg_relevance']:.4f} quality={stats['avg_answer_quality']:.4f}")
            break
        
        time.sleep(10)

if __name__ == "__main__":
    main()
