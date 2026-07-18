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
    
    target_jobs = []
    for j in active_jobs:
        # Target ONLY jobs that are resuming PPO-LSTM (contains --resume-dir and PPO-LSTM)
        if "--resume-dir" in j.entrypoint and "PPO-LSTM" in j.entrypoint:
            target_jobs.append(j)
        else:
            print(f"[KEEPING] Protecting active job: {j.job_id} ({j.entrypoint})")
            
    if not target_jobs:
        print("No orphaned resume jobs found to stop.")
        sys.exit(0)
        
    print(f"\nFound {len(target_jobs)} orphaned PPO-LSTM resume jobs. Stopping them now...\n")
    for job in target_jobs:
        print(f"Stopping Job ID: {job.job_id} ({job.entrypoint})...")
        try:
            client.stop_job(job.job_id)
            print(f"  [SUCCESS] Stop request sent for Job ID: {job.job_id}")
        except Exception as je:
            print(f"  [ERROR] Failed to stop Job ID: {job.job_id}: {je}")
            
    print("\n[COMPLETE] Cleaned up only the orphaned PPO-LSTM resume jobs.")
except Exception as e:
    print(f"[ERROR] Failed to connect or stop jobs: {e}")
