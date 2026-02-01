import numpy as np
from tez_reporter import TezReporter
from config import UAVConfig

TezReporter("physics.py", "Fizik Motoru Yüklendi")

def calculate_path_loss(d: float, fc: float = UAVConfig.FC, eta: float = UAVConfig.ETA) -> float:
    """
    Kanal Kazancı (beta_0) hesabı.
    Tez Önerisi Denklem: beta_0 = (1/eta) * ((4 * pi * fc) / c)^(-2) 
    *Not: Formüldeki eta genellikle paydada olmaz, path loss exponent 'd' üssüdür. 
    Ancak kullanıcı girdisi: Kanal Kazancı (beta_0): (1/eta) * ((4 pi f_c) / c)^-2 şeklinde.
    Burada beta_0, referans mesafedeki (d0=1m) kayıp sabiti olabilir.
    Biz genel friis formülünün d^eta ile ölçeklenmiş halini kullanacağız.
    
    Model: P_rx = P_tx * beta_0 / d^2 (Kullanıcı girdisi d^2 dediği için eta=2 varsayılmış)
    Biz direkt beta_0 hesaplayalım.
    """
    c = UAVConfig.C
    # wavelength lambda
    lam = c / fc
    # Free space path loss at 1m (Friis)
    # FSPL = (4 * pi * d / lambda)^2
    # Gain = 1/FSPL
    # Kullanıcı formülü: ((4 * pi * f_c) / c)^(-2) -> (lambda / 4pi)^2 -> Bu standart Friis.
    # Başındaki (1/eta) katsayısı kullanıcı isteği, aynen ekliyorum.
    
    beta_0 = (1 / eta) * ( (4 * np.pi * fc) / c )**(-2)
    return beta_0

def calculate_received_power(p_tx: float, d: float, beta_0: float) -> float:
    """
    Alınan Güç (P_rx) = P_tx * beta_0 / d^2
    d=0 durumunu engellemek için d en az 1m alınır.
    """
    d = max(d, 1.0)
    return p_tx * beta_0 / (d**2)

def calculate_sinr(p_rx: float, noise_power: float, jamming_power: float) -> float:
    """
    SINR = P_rx / (N0 + I_jam)
    """
    interference = noise_power + jamming_power
    if interference <= 0:
        return np.inf # Teorik sonsuz
    return p_rx / interference

def calculate_data_rate(bandwidth: float, sinr: float) -> float:
    """
    Shannon Kapasitesi: R = B * log2(1 + SINR)
    """
    return bandwidth * np.log2(1 + sinr)

def calculate_flight_power(v: float) -> float:
    """
    İHA Anlık Uçuş Gücü (P_fly)
    P_fly(v) = P0(1 + 3v^2/U_tip^2) + P_ind(v0/v) + 0.5 * d0 * rho * s * A * v^3
    Not: v=0 durumunda P_ind(v0/v) sonsuza gider. 
    Genellikle hover için bu terim P_ind olarak kalır veya v yerine (v^2 + v0^2)^0.5 yaklaşımı kullanılır.
    Ancak kullanıcı denklemi verdi. Biz v < 0.1 ise v=v0 varsayalım veya hover gücünü direkt döndürelim.
    Kullanıcı ayrıca Havada Kalma Gücü (P_hover) = P0 + P_ind vermiş.
    """
    cfg = UAVConfig
    
    if np.abs(v) < 0.1:
        return cfg.P0 + cfg.P_IND

    term1 = cfg.P0 * (1 + (3 * v**2) / (cfg.U_TIP**2))
    # v0/v teriminde v çok küçükse patlar. Fiziksel olarak indüklenmiş hız ileri hız arttıkça düşer.
    # Basitlik ve kararlılık için max(v, 0.1) alabiliriz.
    term2 = cfg.P_IND * (cfg.V0 / v)
    term3 = 0.5 * cfg.D0 * cfg.RHO * cfg.S * cfg.A * (v**3)
    
    return term1 + term2 + term3

def calculate_iot_energy(rate: float) -> float:
    """
    IoT Düğüm Enerji Modeli:
    E_total = E_acq + P_tx * (L_p / R) + P_enc * (L_p * t_enc)
    """
    cfg = UAVConfig
    
    # R (rate) 0 ise iletim süresi sonsuz olur, enerji patlar.
    if rate <= 1e-9:
        transmission_energy = 0 # İletim başarısız, enerji harcanmadı (veya timeout kadar)
        # Amaç modellemek, sonsuz dönmeyelim.
        transmission_time = 10.0 # Timeout
        transmission_energy = cfg.P_TX_NODE * transmission_time
    else:
        transmission_time = cfg.L_P / rate
        transmission_energy = cfg.P_TX_NODE * transmission_time
        
    encoding_energy = cfg.P_ENC * (cfg.L_P * cfg.T_ENC)
    
    return cfg.E_ACQ + transmission_energy + encoding_energy
