import requests, time

BASE = "http://localhost:8000"

# Verify dataset
r = requests.get(f"{BASE}/api/experiments/dataset")
d = r.json()
print("Dataset count:", d['count'])
print("First query:", d['queries'][0]['query'][:60])

# Start experiment
print("\nStarting SQuAD experiment...")
result = requests.post(f"{BASE}/api/experiments/run").json()
print("Result:", result)

# Monitor
print("\nMonitoring:")
last = -1
while True:
    s = requests.get(f"{BASE}/api/experiments/status").json()
    curr = s['current']
    if curr != last:
        pct = (curr/s['total']*100) if s['total'] else 0
        print(f"  [{s['status']}] {curr}/{s['total']} ({pct:.1f}%)")
        last = curr
    if s['status'] == 'completed' or (not s['is_running'] and s['status'] != 'running'):
        print("\nDone!")
        # Get metrics
        m = requests.get(f"{BASE}/api/experiments/metrics").json()
        print("Total runs:", m.get('total_runs'))
        fd = m.get('experiment_1_fault_detection')
        if fd:
            print("Macro F1:", fd.get('macro', {}).get('f1'))
        sc = m.get('experiment_2_strategy_comparison')
        if sc:
            for name, stats in sc.get('strategies', {}).items():
                print(f"{name}: rel={stats['avg_relevance']:.4f} cov={stats['avg_coverage']:.4f} div={stats['avg_diversity']:.4f} qual={stats['avg_answer_quality']:.4f}")
        break
    time.sleep(10)
