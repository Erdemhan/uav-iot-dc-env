import os
import sys

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

print("Connecting to Ray Job Submission Client...")
try:
    from ray.dashboard.modules.job.sdk import JobSubmissionClient
    client = JobSubmissionClient("http://10.70.121.33:8265")
    
    jobs = client.list_jobs()
    active_jobs = [j for j in jobs if j.status in ("RUNNING", "PENDING")]
    
    if not active_jobs:
        print("No active/running jobs found on the cluster.")
        sys.exit(0)
        
    print(f"Found {len(active_jobs)} active jobs. Stopping them now...\n")
    for job in active_jobs:
        print(f"Stopping Job ID: {job.job_id} ({job.entrypoint})...")
        try:
            client.stop_job(job.job_id)
            print(f"  [SUCCESS] Stop request sent for Job ID: {job.job_id}")
        except Exception as je:
            print(f"  [ERROR] Failed to stop Job ID: {job.job_id}: {je}")
            
    print("\n[COMPLETE] All active jobs have been sent stop requests.")
except Exception as e:
    print(f"[ERROR] Failed to connect or stop jobs: {e}")
