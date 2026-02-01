import numpy as np
from tez_reporter import TezReporter
from config import UAVConfig
import physics

TezReporter("entities.py", "Varlık Sınıfları Oluşturuldu")

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
        TransceiverEntity.__init__(self, x, y, z, tx_power=UAVConfig.P_TX_UAV)
        
        self.total_energy_consumed = 0.0

    def consume_energy(self, speed: float, dt: float):
        """
        Physics modülünü kullanarak enerji tüketimini hesaplar ve ekler.
        """
        power = physics.calculate_flight_power(speed)
        energy = power * dt
        self.total_energy_consumed += energy
        return energy

class IoTNode(TransceiverEntity):
    def __init__(self, id: int, x: float, y: float):
        super().__init__(x, y, 0.0, tx_power=UAVConfig.P_TX_NODE)
        self.id = id
        self.aoi = 0.0 # Age of Information
        self.total_energy_consumed = 0.0

    def update_aoi(self, dt: float, success: bool):
        """
        Eğer iletişim başarılıysa AoI sıfırlanır (veya iletim süresi kadar olur),
        değilse geçer zaman kadar artar.
        """
        if success:
            self.aoi = 0.0 # Taze bilgi alındı
        else:
            self.aoi += dt

    def consume_energy(self, rate: float):
        """
        Physics modülünü kullanarak bir paketlik enerji tüketimini hesaplar.
        Rate: Veri hızı (bps)
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
