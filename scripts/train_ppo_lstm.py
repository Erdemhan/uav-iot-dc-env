import sys
import subprocess
import os

if __name__ == "__main__":
    # Backward compatibility wrapper: redirect to unified train.py
    cmd = [sys.executable, "-u", os.path.join(os.path.dirname(__file__), "train.py"), "--algo", "ppo_lstm"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
