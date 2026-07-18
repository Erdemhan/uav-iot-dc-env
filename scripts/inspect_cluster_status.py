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
    
    print("\n=== ACTIVE PLACEMENT GROUPS ===")
    pgs = list_placement_groups()
    active_pgs = [pg for pg in pgs if pg.state == "CREATED"]
    print(f"Total Created Placement Groups: {len(active_pgs)}")
    for pg in active_pgs[:10]:
        print(f"  PG ID: {pg.placement_group_id} | Name: {pg.name} | State: {pg.state} | Bundles: {pg.bundles}")
        
    print("\n=== ACTIVE ACTORS ===")
    actors = list_actors()
    alive_actors = [a for a in actors if a.state == "ALIVE"]
    print(f"Total Alive Actors: {len(alive_actors)}")
    
    # Group by name/class
    actor_counts = {}
    for a in alive_actors:
        cls_name = a.class_name
        actor_counts[cls_name] = actor_counts.get(cls_name, 0) + 1
        
    for cls_name, count in actor_counts.items():
        print(f"  {cls_name}: {count} actors running")
        
except Exception as e:
    print(f"[ERROR] Failed to query cluster status: {e}")
