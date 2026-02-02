import numpy as np

from confs.config import UAVConfig
from confs.env_config import EnvConfig



def calculate_path_loss(d: float, fc: float = UAVConfig.FC, eta: float = UAVConfig.ETA) -> float:
    """
    Channel Gain (beta_0) calculation.
    Thesis Proposal Equation: beta_0 = (1/eta) * ((4 * pi * fc) / c)^(-2) 
    *Note: In the formula, 'eta' is usually not in the denominator; path loss exponent 'eta' is an exponent of 'd'.
    However, user input is: Channel Gain (beta_0): (1/eta) * ((4 pi f_c) / c)^-2.
    Here beta_0 might be the loss constant at reference distance (d0=1m).
    We will use the general Friis formula scaled by d^eta.
    
    Model: P_rx = P_tx * beta_0 / d^2 (Since user input specified d^2, eta=2 is assumed)
    We calculate beta_0 directly.
    """
    c = UAVConfig.C
    # wavelength lambda
    lam = c / fc
    # Free space path loss at 1m (Friis)
    # FSPL = (4 * pi * d / lambda)^2
    # Gain = 1/FSPL
    # User formula: ((4 * pi * f_c) / c)^(-2) -> (lambda / 4pi)^2 -> This is standard Friis.
    # The coefficient (1/eta) at the beginning is a user request, adding it as is.
    
    beta_0 = (1 / eta) * ( (4 * np.pi * fc) / c )**(-2)
    return beta_0

def calculate_received_power(p_tx: float, d: float, beta_0: float) -> float:
    """
    Received Power (P_rx) = P_tx * beta_0 / d^2
    To avoid d=0 issue, d is taken as at least 1m.
    """
    d = np.maximum(d, 1.0)
    return p_tx * beta_0 / (d**2)

def calculate_sinr(p_rx: float, noise_power: float, jamming_power: float) -> float:
    """
    SINR = P_rx / (N0 + I_jam)
    """
    interference = noise_power + jamming_power
    # Vectorized safety check
    # If interference is 0, return inf (handled by numpy widely, but explicit check for scalar existed)
    # We rely on numpy's handling or ensure interference > 0 via N0
    return p_rx / (interference + 1e-30) # Avoid exact zero

def calculate_data_rate(bandwidth: float, sinr: float) -> float:
    """
    Shannon Capacity: R = B * log2(1 + SINR)
    """
    return bandwidth * np.log2(1 + sinr)

def calculate_flight_power(v: float) -> float:
    """
    UAV Instantaneous Flight Power (P_fly)
    P_fly(v) = P0(1 + 3v^2/U_tip^2) + P_ind(v0/v) + 0.5 * d0 * rho * s * A * v^3
    Note: if v=0, P_ind(v0/v) goes to infinity. 
    Generally for hover, this term remains P_ind or (v^2 + v0^2)^0.5 approximation is used.
    However, the user provided the equation. We assume v=v0 if v < 0.1 or return hover power directly.
    User also provided Hovering Power (P_hover) = P0 + P_ind.
    """
    cfg = UAVConfig
    
    if np.abs(v) < 0.1:
        return cfg.P0 + cfg.P_IND

    term1 = cfg.P0 * (1 + (3 * v**2) / (cfg.U_TIP**2))
    # If v is very small in v0/v term, it blows up. Physically, induced velocity drops as forward speed increases.
    # For simplicity and stability, we can take max(v, 0.1).
    term2 = cfg.P_IND * (cfg.V0 / v)
    term3 = 0.5 * cfg.D0 * cfg.RHO * cfg.S * cfg.A * (v**3)
    
    return term1 + term2 + term3

def calculate_iot_energy(rate: float) -> float:
    """
    IoT Node Energy Model:
    E_total = E_acq + P_tx * (L_p / R) + P_enc * (L_p * t_enc)
    """
    cfg = UAVConfig
    env_cfg = EnvConfig
    
    # If R (rate) is 0, transmission time becomes infinite, energy blows up.
    if rate <= 1e-9:
        transmission_energy = 0 # Transmission failed, energy not consumed (or up to timeout)
        # Goal is modeling, avoiding infinite loop.
        transmission_time = 10.0 # Timeout
        transmission_energy = env_cfg.P_TX_NODE * transmission_time
    else:
        transmission_time = cfg.L_P / rate
        transmission_energy = env_cfg.P_TX_NODE * transmission_time
        
    encoding_energy = cfg.P_ENC * (cfg.L_P * cfg.T_ENC)
    
    return cfg.E_ACQ + transmission_energy + encoding_energy
