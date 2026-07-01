"""
Reachable-normalized loss decomposition — ayrı grafikler (S1 ve S2).
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})

def load(run_dir):
    path = os.path.join(run_dir, "comparison", "reachable_norm_results.json")
    with open(path) as f:
        return json.load(f)

S1_DATA = load("artifacts/2026-06-25_10-11-44")
S2_DATA = load("artifacts/2026-06-25_13-56-39")

ALGOS  = ["Baseline", "PPO", "DQN", "PPO-LSTM"]
LABELS = ["QJC\n(Baseline)", "PPO", "DQN", "PPO-LSTM"]

C_JSR = "#2ecc71"
C_PWR = "#e67e22"
C_CH  = "#e74c3c"
ALPHA = 0.90

SCENARIOS = [
    (S1_DATA, "Scenario 1",
     "500 m \u00d7 500 m \u2014 15 Nodes \u2014 1 UAV",
     "artifacts/2026-06-25_10-11-44/comparison/loss_decomposition_s1.png"),
    (S2_DATA, "Scenario 2",
     "1000 m \u00d7 1000 m \u2014 30 Nodes \u2014 2 UAVs",
     "artifacts/2026-06-25_13-56-39/comparison/loss_decomposition_s2.png"),
]

for data, scen_name, subtitle, out_path in SCENARIOS:
    fig, ax = plt.subplots(figsize=(9, 6))

    jsr_m = [np.mean(data[a]["JSR"])         for a in ALGOS]
    pwr_m = [np.mean(data[a]["Power_Gap"])   for a in ALGOS]
    ch_m  = [np.mean(data[a]["Channel_Gap"]) for a in ALGOS]
    jsr_s = [np.std(data[a]["JSR"])          for a in ALGOS]
    pwr_s = [np.std(data[a]["Power_Gap"])    for a in ALGOS]
    ch_s  = [np.std(data[a]["Channel_Gap"])  for a in ALGOS]

    x = np.arange(len(ALGOS))
    w = 0.55

    # Stacked bars
    ax.bar(x, ch_m,  w, color=C_CH,  alpha=ALPHA, label="Channel Loss")
    ax.bar(x, pwr_m, w, color=C_PWR, alpha=ALPHA, label="Power Loss",
           bottom=ch_m)
    bottom_jsr = [c + p for c, p in zip(ch_m, pwr_m)]
    ax.bar(x, jsr_m, w, color=C_JSR, alpha=ALPHA, label="JSR (Success)",
           bottom=bottom_jsr)

    # Hata çubukları (JSR için)
    ax.errorbar(x, [b + j for b, j in zip(bottom_jsr, jsr_m)],
                yerr=jsr_s, fmt="none", color="#1a8a4a", capsize=5, lw=1.5, capthick=1.5)

    # Değer etiketleri
    for i, (c, p, j, js) in enumerate(zip(ch_m, pwr_m, jsr_m, jsr_s)):
        # Kanal kaybı etiketi
        if c >= 5:
            ax.text(x[i], c / 2, f"{c:.1f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
        # Güç kaybı etiketi
        if p >= 5:
            ax.text(x[i], c + p / 2, f"{p:.1f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
        # JSR etiketi (bar içi)
        if j >= 5:
            ax.text(x[i], c + p + j / 2, f"{j:.1f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
        # JSR ± (bar üstü)
        top = c + p + j
        ax.text(x[i], top + 2.5, f"{j:.1f} ± {js:.1f}%",
                ha="center", va="bottom", fontsize=9,
                color="#1a8a4a", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Percentage of Reachable Steps (%)", fontsize=12)
    ax.set_title(f"{scen_name}\n{subtitle}", fontsize=13, fontweight="bold", pad=10)
    ax.axhline(100, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    handles = [
        mpatches.Patch(color=C_JSR, alpha=ALPHA, label="JSR (Jamming Success Rate)"),
        mpatches.Patch(color=C_PWR, alpha=ALPHA, label="Power Loss (right channel, dist. too large)"),
        mpatches.Patch(color=C_CH,  alpha=ALPHA, label="Channel Loss (wrong channel)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path}")
    plt.close()
