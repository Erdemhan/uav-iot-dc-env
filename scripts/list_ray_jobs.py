import ray
import os
import sys

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

print("Connecting to Ray cluster...")
try:
    ray.init(address="ray://10.70.121.33:10001", ignore_reinit_error=True)
    print("[SUCCESS] Connected to Ray!")
    
    from ray._private.state import state
    print("\n=== ACTIVE RAY JOBS ===")
    jobs = state.jobs()
    for job in jobs:
        print(f"Job ID: {job.get('JobID')} | Status: {job.get('Status')} | Driver IP: {job.get('DriverIPAddress')} | Start Time: {job.get('StartTime')}")
        
except Exception as e:
    print(f"[ERROR] Failed to fetch Ray state: {e}")
