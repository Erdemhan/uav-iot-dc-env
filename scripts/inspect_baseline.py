
import numpy as np
import os

path = "artifacts/2026-02-10_21-07-45/baseline"
q_file = os.path.join(path, "q_table.npy")
c_file = os.path.join(path, "counts.npy")

if os.path.exists(q_file):
    q_table = np.load(q_file)
    print("Q-Table:", q_table)
    print("Argmax Action:", np.argmax(q_table))
else:
    print("Q-Table not found")

if os.path.exists(c_file):
    counts = np.load(c_file)
    print("Counts:", counts)
    print("Total Steps:", np.sum(counts))
