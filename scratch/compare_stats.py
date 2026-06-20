import json
import numpy as np

file1 = r"E:\Projeler\TEZ\uav-iot-dc-env\artifacts\2026-06-20_17-44-40\comparison\robustness_results_30seeds.json"
file2 = r"E:\Projeler\TEZ\uav-iot-dc-env\artifacts\2026-06-20_20-28-20\comparison\robustness_results_30seeds.json"

def print_stats(file_path, label):
    print(f"\n=== {label} ===")
    with open(file_path, 'r') as f:
        data = json.load(f)
    for algo, metrics in data.items():
        print(f"Algorithm: {algo}")
        for metric, values in metrics.items():
            mean = np.mean(values)
            std = np.std(values)
            print(f"  {metric}: {mean:.2f}% ± {std:.2f}%" if "%" not in metric and metric in ["JSR", "Tracking_Acc"] else f"  {metric}: {mean:.4f} ± {std:.4f}")

print_stats(file1, "Run 1 (LR=5e-5, Gamma=0.95)")
print_stats(file2, "Run 2 (LR=1e-4, Gamma=0.90)")
