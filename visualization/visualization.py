import matplotlib.pyplot as plt
import matplotlib.patches as patches

from core.config import UAVConfig
from core.env_config import EnvConfig



class Visualization:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.area_size = EnvConfig.AREA_SIZE
        plt.ion() # Interactive mode

    def render(self, env):
        self.ax.clear()
        
        # Area bounds
        self.ax.set_xlim(0, self.area_size)
        self.ax.set_ylim(0, self.area_size)
        
        # UAV
        self.ax.scatter(env.uav.x, env.uav.y, c='blue', marker='^', s=100, label='UAV')
        
        # Nodes
        node_x = [n.x for n in env.nodes]
        node_y = [n.y for n in env.nodes]
        self.ax.scatter(node_x, node_y, c='green', marker='o', s=50, label='IoT Node')
        
        # Attacker
        self.ax.scatter(env.attacker.x, env.attacker.y, c='red', marker='x', s=80, label='Attacker')
        
        # Jamming Effet (Visual ring)
        # Draw if power > 0
        if env.attacker.jamming_power > 0.1:
            # Radius proportional to power (e.g. power * 100m)
            radius = env.attacker.jamming_power * 100 
            circle = patches.Circle((env.attacker.x, env.attacker.y), radius, 
                                    edgecolor='red', facecolor='red', alpha=0.1)
            self.ax.add_patch(circle)
            self.ax.text(env.attacker.x, env.attacker.y + radius, "Jamming!", color='red')

        # Info Screen
        info_text = f"Step: {env.current_step}\n"
        info_text += f"Jammer Power: {env.attacker.jamming_power:.2f} W\n"
        avg_sinr = sum([10*1.0 for n in env.nodes]) / len(env.nodes) # Placeholder
        # Real SINR could be retrieved from environment but visual is enough for now.
        
        self.ax.legend(loc='upper right')
        self.ax.set_title("IoT Based UAV Application - Security Simulation")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.draw()
        plt.pause(0.01) # Wait for UI update
