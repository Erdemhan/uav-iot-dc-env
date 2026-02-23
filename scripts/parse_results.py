
import json
import numpy as np
import os

def main():
    path = "paper/robustness_results_30seeds.json"
    if not os.path.exists(path):
        print("File not found")
        return

    with open(path, "r") as f:
        data = json.load(f)

    print(f"{'Algorithm':<15} | {'JSR (%)':<15} | {'Track (%)':<15} | {'Power (W)':<15} | {'SINR (dB)':<15}")
    print("-" * 80)

    for algo, metrics in data.items():
        jsr = np.array(metrics["JSR"])
        track = np.array(metrics["Tracking_Acc"])
        power = np.array(metrics["Power"])
        sinr = np.array(metrics["SINR"])

        print(f"{algo:<15} | {np.mean(jsr):.1f} +/- {np.std(jsr):.1f} | "
              f"{np.mean(track):.1f} +/- {np.std(track):.1f} | "
              f"{np.mean(power):.3f} +/- {np.std(power):.3f} | "
              f"{np.mean(sinr):.2f} +/- {np.std(sinr):.2f}")

if __name__ == "__main__":
    main()
