import numpy as np

from confs.config import UAVConfig
from confs.env_config import EnvConfig
import core.physics as physics



class BaseEntity:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    @property
    def position(self):
        return np.array([self.x, self.y, self.z])

class MobileEntity(BaseEntity):
    def __init__(self, x: float, y: float, z: float = 0.0):
        BaseEntity.__init__(self, x, y, z)
        self.vx = 0.0
        self.vy = 0.0

    def move(self, dx: float, dy: float):
        self.x += dx
        self.y += dy

    @property
    def velocity_magnitude(self) -> float:
        return np.sqrt(self.vx**2 + self.vy**2)

class TransceiverEntity(BaseEntity):
    def __init__(self, x: float, y: float, z: float, tx_power: float):
        BaseEntity.__init__(self, x, y, z)
        self.tx_power = tx_power

class UAVAgent(MobileEntity, TransceiverEntity):
    def __init__(self, x: float, y: float, z: float):
        # Explicit calls to parent inits
        MobileEntity.__init__(self, x, y, z)
        TransceiverEntity.__init__(self, x, y, z, tx_power=EnvConfig.P_TX_UAV)
        
        self.total_energy_consumed = 0.0
        self.current_channel = 0 # Default Channel Index

    def consume_energy(self, speed: float, dt: float):
        """
        Calculates and adds energy consumption using the Physics module.
        """
        power = physics.calculate_flight_power(speed)
        energy = power * dt
        self.total_energy_consumed += energy
        return energy

class IoTNode(TransceiverEntity):
    def __init__(self, id: int, x: float, y: float):
        super().__init__(x, y, 0.0, tx_power=EnvConfig.P_TX_NODE)
        self.id = id
        self.aoi = 0.0 # Age of Information
        self.total_energy_consumed = 0.0
        # Status: 0=Connected, 1=Out of Range, 2=Jammed
        self.connection_status = 0
        self.current_channel = 0 # Nodes usually follow UAV or fixed, assumed synchronized for now
        
        # Advanced Metrics
        self.current_connected_duration = 0.0
        self.total_connected_duration = 0.0
        self.max_continuous_duration = 0.0

    def update_aoi(self, dt: float, success: bool):
        """
        Updates Age of Information and Connection Stats.
        """
        if success:
            self.aoi = 0.0 # Fresh information received
            
            # Stats
            self.current_connected_duration += dt
            self.total_connected_duration += dt
            if self.current_connected_duration > self.max_continuous_duration:
                self.max_continuous_duration = self.current_connected_duration
        else:
            self.aoi += dt
            self.current_connected_duration = 0.0 # Reset streak

    def consume_energy(self, rate: float):
        """
        Calculates the energy consumption for one packet using the Physics module.
        Rate: Data rate (bps)
        """
        e_packet = physics.calculate_iot_energy(rate)
        self.total_energy_consumed += e_packet
        return e_packet

class SmartAttacker(BaseEntity):
    def __init__(self, x: float, y: float):
        super().__init__(x, y, 0.0)
        self.jamming_power = 0.0
        self.current_channel = 0
        
        # QJC Algo State (Liao et al.)
        self.num_channels = len(UAVConfig.CHANNELS)
        self.q_table = np.zeros(self.num_channels)
        self.channel_counts = np.zeros(self.num_channels) # mu
        
        # Hyperparams from Config
        from confs.model_config import QJCConfig
        self.tau_0 = QJCConfig.TAU_0
        self.gamma = QJCConfig.GAMMA
        self.temp_xi = QJCConfig.TEMP_XI

    def set_jamming_power(self, power: float):
        self.jamming_power = max(0.0, power)

    def select_channel_qjc(self) -> int:
        """
        Softmax Action Selection (Eq. 22 Liao et al.)
        psi_i = exp(Q_i / xi) / sum(...)
        """
        if self.temp_xi <= 0:
            return np.argmax(self.q_table) # Greedy fallback
            
        exps = np.exp(self.q_table / self.temp_xi)
        probs = exps / np.sum(exps)
        
        action = np.random.choice(np.arange(self.num_channels), p=probs)
        return action

    def update_qjc(self, channel: int, reward: float):
        """
        Q-Update Rule (Eq. 24 Liao et al.)
        Q(k) = (1 - tau) * Q(k) + tau * (L_j + gamma * max(Q))
        tau = tau_0 / (mu * log10(mu + 1.1))
        """
        self.channel_counts[channel] += 1
        mu = self.channel_counts[channel]
        
        # Adaptive learning rate
        from confs.model_config import QJCConfig
        tau = self.tau_0 / (mu * np.log10(mu + QJCConfig.MU_OFFSET))
        
        # Update
        # Note: Liao's Algo 2 uses max(Q_next) which implies we need next state optimal.
        # Since this is a single-state (or state-less channel selection) Bandit-like context in this function,
        # max(Q) usually refers to max Q of current table for next step estimation or standard Q-learning.
        # Given the paper context, we assume standard Q-learning update where max_q is max(Q) of current table (Target).
        max_q = np.max(self.q_table) 
        
        self.q_table[channel] = (1 - tau) * self.q_table[channel] + tau * (reward + self.gamma * max_q)

    def save_model(self, path="baseline_q_table"):
        """Saves Q-Table and Counts."""
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(f"{path}/q_table.npy", self.q_table)
        np.save(f"{path}/counts.npy", self.channel_counts)
        
    def load_model(self, path="baseline_q_table"):
        """Loads Q-Table and Counts."""
        import os
        try:
            self.q_table = np.load(f"{path}/q_table.npy")
            self.channel_counts = np.load(f"{path}/counts.npy")
            print(f"Baseline Q-Table loaded from {path}")
        except FileNotFoundError:
            print(f"No saved Q-Table found at {path}. Starting fresh.")
