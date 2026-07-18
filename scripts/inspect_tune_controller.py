import inspect
import os
import sys

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

print("Inspecting Ray's TuneController...")
try:
    from ray.tune.execution.tune_controller import TuneController
    print("\n[SUCCESS] Imported TuneController!")
    
    print("\n=== METHODS OF TuneController ===")
    methods = [name for name, member in inspect.getmembers(TuneController, predicate=inspect.isfunction)]
    for m in sorted(methods):
        print(f"  {m}")
        
except Exception as e:
    print(f"[ERROR] Failed to inspect TuneController: {e}")
