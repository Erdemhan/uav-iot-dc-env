import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

from tez_reporter import TezReporter
from config import UAVConfig
from entities import UAVAgent, IoTNode, SmartAttacker
import physics
from logger import SimulationLogger

TezReporter("environment.py", "UAV_IoT_Env Ortamı Oluşturuldu")

class UAV_IoT_Env(gym.Env):
    """
    OpenAI Gymnasium uyumlu UAV-IoT Simülasyon Ortamı.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, logger: SimulationLogger = None):
        super().__init__()
        
        self.logger = logger
        
        # Alan sınırları
        self.area_size = UAVConfig.AREA_SIZE
        
        # Ajanların oluşturulması
        self.uav = UAVAgent(x=self.area_size/2, y=self.area_size/2, z=UAVConfig.H)
        
        self.nodes = []
        for i in range(UAVConfig.NUM_NODES):
            # Rastgele konumlandır
            nx = np.random.uniform(0, self.area_size)
            ny = np.random.uniform(0, self.area_size)
            self.nodes.append(IoTNode(i, nx, ny))
            
        self.attacker = SmartAttacker(x=self.area_size/2 + 100, y=self.area_size/2 + 100) # Sabit konum (şimdilik)
        
        # Action Space: Attacker Jamming Power (Sürekli)
        # Örnek: 0 ile 2 Watt arası jamming
        self.action_space = spaces.Box(low=0.0, high=0.5, shape=(1,), dtype=np.float32)
        
        # Observation Space:
        # UAV (x, y), Attacker (x, y), Nodes (x, y) * N, Node SINR * N, Node AoI * N
        # Toplam boyut: 2 + 2 + 2*N + N + N = 4 + 4*N
        obs_dim = 4 + 4 * UAVConfig.NUM_NODES
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.current_step = 0
        self.max_steps = 100 # Bölüm başına adım sayısı
        self.dt = 5.0 # Zaman adımı (saniye)
        
        # Dairesel hareket için açı
        self.uav_angle = 15.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.uav_angle = 0.0
        
        # UAV merkeze al
        self.uav.x = self.area_size/2
        self.uav.y = self.area_size/2
        self.uav.total_energy_consumed = 0.0
        
        # Nodeları sıfırla
        for node in self.nodes:
            node.aoi = 0.0
            node.total_energy_consumed = 0.0
            
        self.attacker.set_jamming_power(0.0)
        
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        obs.extend([self.uav.x, self.uav.y])
        obs.extend([self.attacker.x, self.attacker.y])
        
        sinrs = []
        aois = []
        for node in self.nodes:
            obs.extend([node.x, node.y])
            aois.append(node.aoi)
            # SINR anlık hesaplanmalı, burada son bilinen değer saklanabilir ama
            # step içinde hesaplayıp state'e koymak daha doğru. 
            # Şimdilik 0 placeholder koyuyoruz, step fonksiyonunda güncel döner.
            sinrs.append(0.0) 
            
        obs.extend(sinrs)
        obs.extend(aois)
        
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        
        # 1. Aksiyon Uygula (Attacker)
        jam_power = float(action[0])
        self.attacker.set_jamming_power(jam_power)
        
        # 2. UAV Hareketi (Basit Dairesel Yörünge)
        # Merkez etrafında r=200m yarıçaplı daire
        radius = 200
        center_x, center_y = self.area_size/2, self.area_size/2
        speed = 10.0 # m/s (Örnek hız)
        
        # Açısal hız w = v / r
        angular_speed = speed / radius
        self.uav_angle += angular_speed * self.dt
        
        new_x = center_x + radius * np.cos(self.uav_angle)
        new_y = center_y + radius * np.sin(self.uav_angle)
        
        # Hız vektörü güncelle
        self.uav.vx = (new_x - self.uav.x) / self.dt
        self.uav.vy = (new_y - self.uav.y) / self.dt
        
        self.uav.x = new_x
        self.uav.y = new_y
        
        # UAV Enerji Tüketimi
        v_uav = self.uav.velocity_magnitude
        self.uav.consume_energy(v_uav, self.dt)
        
        # 3. Fizik Hesaplamaları (Her bir node için)
        total_sinr = 0
        jammed_count = 0
        step_log = {
            "step": self.current_step,
            "uav_x": self.uav.x,
            "uav_y": self.uav.y,
            "jammer_power": jam_power,
            "uav_energy": self.uav.total_energy_consumed
        }
        
        for i, node in enumerate(self.nodes):
            # Mesafe
            dist_uav = np.linalg.norm(node.position - self.uav.position)
            dist_jam = np.linalg.norm(node.position - self.attacker.position)
            
            # Kanal Kazançları
            beta_uav = physics.calculate_path_loss(dist_uav)
            # Jammer yer seviyesinde ise, jammer'dan node'a:
            # Basitlik için aynı path loss modelini kullanalım (veya farklı olabilir)
            beta_jam = physics.calculate_path_loss(dist_jam)
            
            # Güçler
            p_rx_signal = physics.calculate_received_power(node.tx_power, dist_uav, beta_uav)
            p_rx_jam = physics.calculate_received_power(self.attacker.jamming_power, dist_jam, beta_jam)
            
            # SINR
            sinr = physics.calculate_sinr(p_rx_signal, UAVConfig.N0_Linear, p_rx_jam)
            total_sinr += sinr
            
            # Rate
            rate = physics.calculate_data_rate(UAVConfig.B, sinr)
            
            # Bağlantı Durumu (Eşik SINR dB mesela 0 dB -> Linear 1.0)
            is_connected = sinr > 1.0 
            if not is_connected:
                jammed_count += 1
                
            # Node Durum Güncelle
            node.update_aoi(self.dt, success=is_connected)
            node.consume_energy(rate)
            
            # Log Individual Node Metrics
            step_log[f"node_{i}_x"] = node.x
            step_log[f"node_{i}_y"] = node.y
            step_log[f"node_{i}_sinr"] = sinr
            step_log[f"node_{i}_aoi"] = node.aoi
            step_log[f"node_{i}_energy"] = node.total_energy_consumed
            step_log[f"node_{i}_connected"] = 1 if is_connected else 0

        # 4. Ödül: Saldırganın amacı iletişimi kesmek (+1 jammed node başına)
        reward = jammed_count
        
        # 5. Loglama
        step_log["reward"] = reward
        step_log["jammed_count"] = jammed_count
        
        if self.logger:
            self.logger.log_step(step_log)
            
        # Bitiş kontrolü
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = step_log
        
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        # Render visualization.py tarafından yapılacak, state döndürmek yeterli olabilir
        # veya Visualization sınıfı buraya parametre olarak geçilebilir.
        # Main dosyasında Visualization.render(env) çağrısı yapılacak.
        pass
