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

    def set_jamming_power(self, power: float):
        self.jamming_power = max(0.0, power)
