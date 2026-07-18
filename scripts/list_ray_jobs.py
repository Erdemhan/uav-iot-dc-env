import ray
import os
import sys

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

print("Connecting to Ray cluster...")
try:
    # Use ray.init client connection
    ray.init(address="ray://10.70.121.33:10001", ignore_reinit_error=True)
    print("[SUCCESS] Connected to Ray!")
    
    from ray.util.state import list_jobs
    print("\n=== ACTIVE RAY JOBS ===")
    jobs = list_jobs()
    if not jobs:
        print("No active jobs found in the cluster.")
    for job in jobs:
        print(f"Job ID: {job.job_id} | Status: {job.status} | Entrypoint: {job.entrypoint} | Start Time: {job.start_time}")
        
except Exception as e:
    print(f"[ERROR] Failed to fetch Ray state: {e}")
