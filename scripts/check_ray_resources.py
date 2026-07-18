import ray
import os
import sys

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

print("Connecting to Ray Cluster...")
try:
    ray.init(address="auto")
    print("\n[SUCCESS] Connected to Ray cluster!")
    
    print("\n=== TOTAL CLUSTER RESOURCES ===")
    resources = ray.cluster_resources()
    for k, v in sorted(resources.items()):
        print(f"  {k}: {v}")
        
    print("\n=== AVAILABLE (FREE) RESOURCES ===")
    avail = ray.available_resources()
    for k, v in sorted(avail.items()):
        print(f"  {k}: {v}")
        
    print("\n=== ACTIVE NODES ===")
    nodes = ray.nodes()
    print(f"Number of nodes: {len(nodes)}")
    for idx, node in enumerate(nodes):
        print(f"  Node {idx}: IP={node.get('NodeManagerAddress')}, Alive={node.get('Alive')}, Resources={node.get('Resources')}")
        
except Exception as e:
    print(f"[ERROR] Failed to check Ray resources: {e}")
