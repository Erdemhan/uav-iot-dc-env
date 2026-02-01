import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tez_reporter import TezReporter
from config import UAVConfig

TezReporter("visualization.py", "Görselleştirme Modülü Hazırlandı")

class Visualization:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.area_size = UAVConfig.AREA_SIZE
        plt.ion() # Etkileşimli mod

    def render(self, env):
        self.ax.clear()
        
        # Alan sınırları
        self.ax.set_xlim(0, self.area_size)
        self.ax.set_ylim(0, self.area_size)
        
        # UAV
        self.ax.scatter(env.uav.x, env.uav.y, c='blue', marker='^', s=100, label='İHA')
        
        # Nodes
        node_x = [n.x for n in env.nodes]
        node_y = [n.y for n in env.nodes]
        self.ax.scatter(node_x, node_y, c='green', marker='o', s=50, label='IoT Düğümü')
        
        # Attacker
        self.ax.scatter(env.attacker.x, env.attacker.y, c='red', marker='x', s=80, label='Saldırgan')
        
        # Jamming Etkisi (Görsel halka)
        # Güç 0'dan büyükse çiz
        if env.attacker.jamming_power > 0.1:
            # Yarıçapı güçle orantılı temsili çizelim (örneğin power * 100m)
            radius = env.attacker.jamming_power * 100 
            circle = patches.Circle((env.attacker.x, env.attacker.y), radius, 
                                    edgecolor='red', facecolor='red', alpha=0.1)
            self.ax.add_patch(circle)
            self.ax.text(env.attacker.x, env.attacker.y + radius, "Jamming!", color='red')

        # Bilgi Ekranı
        info_text = f"Step: {env.current_step}\n"
        info_text += f"Jammer Power: {env.attacker.jamming_power:.2f} W\n"
        avg_sinr = sum([10*1.0 for n in env.nodes]) / len(env.nodes) # Placeholder
        # Gerçek SINR'ı environment'tan alabilirsek iyi olur ama şimdilik görsel yeterli.
        
        self.ax.legend(loc='upper right')
        self.ax.set_title("IoT Tabanlı İHA Uygulaması - Güvenlik Simülasyonu")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.draw()
        plt.pause(0.01) # UI güncellenmesi için bekle
