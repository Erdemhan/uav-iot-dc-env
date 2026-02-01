import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tez_reporter import TezReporter

TezReporter("visualizer.py", "Görselleştirme Modülü Başlatıldı")

class SimulationVisualizer:
    def __init__(self, exp_dir: str = None):
        if exp_dir is None:
            exp_dir = self.find_latest_experiment()
            
        self.exp_dir = exp_dir
        self.csv_path = os.path.join(exp_dir, "history.csv")
        self.config_path = os.path.join(exp_dir, "config.json")
        
        # Load Data
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Data Loaded: {len(self.df)} steps.")
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV Not Found: {self.csv_path}")

        # Load Config (for static positions if needed)
        self.config = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.config = json.load(f)

        # Preprocess Data
        self._preprocess_data()
        
        # Style
        sns.set_style("whitegrid")

    def find_latest_experiment(self, logs_dir="logs") -> str:
        """En son tarihli deney klasörünü bulur."""
        all_dirs = glob.glob(os.path.join(logs_dir, "EXP_*"))
        if not all_dirs:
            raise FileNotFoundError("Hiçbir deney kaydı bulunamadı (logs/ boş).")
        latest_dir = max(all_dirs, key=os.path.getmtime)
        return latest_dir

    def _preprocess_data(self):
        """Ham veriyi analiz için hazırlar."""
        # 1. Calculate Average System SINR (dB)
        # Find all columns starting with 'node_' and ending with '_sinr'
        sinr_cols = [c for c in self.df.columns if "node_" in c and "_sinr" in c]
        if sinr_cols:
            # Linear avg
            avg_linear_sinr = self.df[sinr_cols].mean(axis=1)
            # Avoid log(0) error
            avg_linear_sinr = avg_linear_sinr.replace(0, 1e-12) 
            self.df["sinr_db"] = 10 * np.log10(avg_linear_sinr)
        else:
            self.df["sinr_db"] = 0.0

        # 2. Calculate Average AoI
        aoi_cols = [c for c in self.df.columns if "node_" in c and "_aoi" in c]
        if aoi_cols:
            self.df["aoi_avg"] = self.df[aoi_cols].mean(axis=1)
        else:
            self.df["aoi_avg"] = 0.0

        # 3. Jammed Status flag
        if "jammed_count" in self.df.columns:
            self.df["is_jammed"] = (self.df["jammed_count"] > 0).astype(int)
        else:
            self.df["is_jammed"] = 0

    def plot_trajectory(self):
        """İHA ve Saldırgan yörüngesini çizer."""
        plt.figure(figsize=(10, 10))
        
        # 1. UAV Path
        plt.plot(self.df["uav_x"], self.df["uav_y"], label="UAV Trajectory", color="blue", linewidth=2)
        
        # 2. Start/End Points
        plt.scatter(self.df["uav_x"].iloc[0], self.df["uav_y"].iloc[0], color="green", s=100, label="Start")
        plt.scatter(self.df["uav_x"].iloc[-1], self.df["uav_y"].iloc[-1], color="blue", marker="s", s=100, label="End")

        # 3. Jammed Events -> Red Dots
        jammed_points = self.df[self.df["is_jammed"] == 1]
        if not jammed_points.empty:
            plt.scatter(jammed_points["uav_x"], jammed_points["uav_y"], 
                        color="red", s=30, zorder=5, label="Jamming Detection")

        # 4. Attacker Position (Static assumed or from config if available)
        # Config'den okumaya çalışalım, yoksa hardcode
        # Environment'te: self.area_size/2 + 100
        area_size = float(self.config.get("AREA_SIZE", 1000))
        att_x = area_size/2 + 100
        att_y = area_size/2 + 100
        plt.scatter(att_x, att_y, color="red", marker="X", s=200, label="Attacker Station")

        # 5. Nodes (Estimated from config count, positions random so can't plot exact without logs)
        # Note: If node positions were not logged per step (only changed if mobile), 
        # normally we should log them once. 
        # But environment logs 'node_i_x' IF we implemented it.
        # Let's check environment again... 
        # Environment.py: `obs.extend([node.x, node.y])` -> Observation has it.
        # But `log_step` dict? 
        # `log_step = {"step":..., "uav_x":..., ...}` 
        # And loop: `step_log[f"node_{i}_sinr"] = ...`
        # It seems I did NOT log node positions in CSV explicitly in loop!
        # So I cannot plot exact node positions unless I re-simulate or parse config if saved there (random seed?).
        # I will skip plotting nodes to avoid misleading info, or plot a placeholder text.
        
        plt.title("UAV Trajectory & Jamming Events", fontsize=14)
        plt.xlabel("Position X (m)")
        plt.ylabel("Position Y (m)")
        plt.legend()
        plt.xlim(0, area_size)
        plt.ylim(0, area_size)
        
        save_path = os.path.join(self.exp_dir, "trajectory.png")
        plt.savefig(save_path, dpi=300)
        print(f"Trajectory saved: {save_path}")
        plt.close()

    def plot_metrics(self):
        """Zaman serisi metriklerini çizer (SINR, AoI, Energy)."""
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        
        steps = self.df["step"]

        # Subplot 1: SINR (dB) vs Jamming Power
        ax1 = axes[0]
        color = 'tab:blue'
        ax1.set_ylabel('Avg SINR (dB)', color=color, fontsize=12)
        ax1.plot(steps, self.df["sinr_db"], color=color, linewidth=2, label="SINR (dB)")
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Threshold Line (Example: 0 dB)
        ax1.axhline(0, color='gray', linestyle='--', label="Threshold (0 dB)")

        # Twin Axis for Jamming
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Jamming Power (W)', color=color, fontsize=12)
        ax2.plot(steps, self.df["jammer_power"], color=color, linestyle=':', linewidth=2, label="Jamming Power")
        ax2.tick_params(axis='y', labelcolor=color)
        
        ax1.set_title("Communication Quality vs Jamming Attack", fontsize=14)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # Subplot 2: Age of Information (AoI)
        ax_aoi = axes[1]
        ax_aoi.plot(steps, self.df["aoi_avg"], color="green", label="Avg AoI")
        ax_aoi.set_ylabel("Avg AoI (s)", fontsize=12)
        ax_aoi.set_title("Information Freshness (Age of Information)", fontsize=14)
        ax_aoi.legend()
        ax_aoi.grid(True, which="both", linestyle='--')

        # Subplot 3: UAV Energy Consumption
        ax_eng = axes[2]
        ax_eng.plot(steps, self.df["uav_energy"], color="orange", linewidth=2, label="Total Energy")
        ax_eng.set_xlabel("Time Step", fontsize=12)
        ax_eng.set_ylabel("Energy (J)", fontsize=12)
        ax_eng.set_title("UAV Total Energy Consumption", fontsize=14)
        ax_eng.legend()

        plt.tight_layout()
        save_path = os.path.join(self.exp_dir, "metrics_analysis.png")
        plt.savefig(save_path, dpi=300)
        print(f"Metrics analysis saved: {save_path}")
        plt.close()

    def generate_report(self):
        """Rapor oluşturmayı tetikler."""
        print(f"--- Visualization Report: {self.exp_dir} ---")
        self.plot_trajectory()
        self.plot_metrics()
        print("Visualization Completed.")

if __name__ == "__main__":
    viz = SimulationVisualizer()
    viz.generate_report()
