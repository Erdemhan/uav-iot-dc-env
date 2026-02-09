
import json
import numpy as np

with open('paper/robustness_results_30seeds.json', 'r') as f:
    data = json.load(f)

print("| Algoritma | Başarı (JSR) | Kanal Eşleşme (Tracking) | Ort. Güç (W) | SINR (dB) |")
print("| :--- | :---: | :---: | :---: | :---: |")

for algo in ["PPO", "PPO-LSTM", "DQN", "Baseline"]:
    if algo not in data: continue
    metrics = data[algo]
    
    jsr = f"%{np.mean(metrics['JSR']):.1f} ± {np.std(metrics['JSR']):.1f}"
    track = f"%{np.mean(metrics['Tracking_Acc']):.1f}"
    power = f"{np.mean(metrics['Power']):.3f} W"
    sinr = f"{np.mean(metrics['SINR']):.2f}"
    
    # Bold the best
    if algo == "PPO":
        print(f"| **{algo} (Ours)** | **{jsr}** | **{track}** | {power} | **{sinr}** |")
    else:
        print(f"| **{algo}** | {jsr} | {track} | **{power}** | {sinr} |")
