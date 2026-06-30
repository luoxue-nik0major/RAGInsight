import requests, time, sys
BASE = "http://localhost:8000"
while True:
    try:
        r = requests.get(f"{BASE}/api/experiments/status", timeout=10)
        d = r.json()
        print(f"[{d['status']}] {d['current']}/{d['total']}")
        if d['status'] == 'completed' or not d['is_running']:
            break
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(8)
print("Done monitoring.")
