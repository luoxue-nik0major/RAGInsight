import requests
d = requests.get('http://localhost:8000/api/experiments/status').json()
print("status:", d['status'])
print("current:", d['current'])
print("total:", d['total'])
print("running:", d['is_running'])
