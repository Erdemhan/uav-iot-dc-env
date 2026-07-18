import ray
import os
import sys

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

print("Connecting to Ray cluster status API...")
try:
    # Use ray.init client connection
    ray.init(address="ray://10.70.121.33:10001", ignore_reinit_error=True)
    print("[SUCCESS] Connected to Ray!")
    
    from ray.util.state import list_actors, list_placement_groups
    
    print("\n=== PLACEMENT GROUPS ===")
    pgs = list_placement_groups()
    print(f"Total Placement Groups: {len(pgs)}")
    
    # Group by state
    pg_states = {}
    for pg in pgs:
        pg_states[pg.state] = pg_states.get(pg.state, 0) + 1
    for state, count in pg_states.items():
        print(f"  State '{state}': {count}")
        
    print("\n=== DETAIL OF ACTIVE & PENDING PLACEMENT GROUPS ===")
    active_pgs = [pg for pg in pgs if pg.state in ("CREATED", "PENDING")]
    for pg in active_pgs:
        print(f"  PG ID: {pg.placement_group_id} | State: {pg.state} | Bundles: {pg.bundles}")
        
    print("\n=== ACTORS ===")
    actors = list_actors()
    print(f"Total Actors: {len(actors)}")
    
    # Group by state
    actor_states = {}
    for a in actors:
        actor_states[a.state] = actor_states.get(a.state, 0) + 1
    for state, count in actor_states.items():
        print(f"  State '{state}': {count}")
        
    print("\n=== DETAIL OF ALIVE & PENDING ACTORS ===")
    active_actors = [a for a in actors if a.state in ("ALIVE", "PENDING_CREATION")]
    for a in active_actors[:15]:
        print(f"  Actor: {a.class_name} | State: {a.state} | Job ID: {a.job_id}")
        
except Exception as e:
    print(f"[ERROR] Failed to query cluster status: {e}")
