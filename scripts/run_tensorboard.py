import sys
import os

# Resolve project root dynamically
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Monkeypatching Tensorboard's port check function to bypass WSL2 localhost socket freezing bug
try:
    import tensorboard.program as tb_program
    tb_program.is_port_in_use = lambda port: False
    print("[Monkeypatch] Successfully bypassed Tensorboard's localhost port check.")
except Exception as e:
    print(f"Warning: Could not apply monkeypatch: {e}")

# Run official Tensorboard main
from tensorboard.main import run_main
if __name__ == "__main__":
    sys.exit(run_main())