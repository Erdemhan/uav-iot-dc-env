



class UAVConfig:
    """
    Thesis proposal Table 3 and default simulation parameters.
    """
    # Visualization settings
    SIMULATION_DELAY = 0 # Seconds
    
    # Physical Constants
    C = 3e8       # Speed of light (m/s)
    G = 9.8       # Gravity (m/s^2)


    # UAV Mechanical Parameters (Rotary Wing)
    H = 100.0     # Operation altitude (m)
    U_TIP = 120.0 # Rotor tip speed (m/s)
    V0 = 4.03     # Mean rotor induced velocity (m/s)
    P0 = 79.86    # Blade profile power (W)
    P_IND = 88.63 # Induced power (W)
    RHO = 1.225   # Air density (kg/m^3)
    D0 = 0.6      # Fuselage drag ratio
    S = 0.05      # Rotor solidity
    A = 0.5       # Rotor disc area (m^2)

    # Communication Parameters
    B = 2e6       # Bandwidth (Hz) -> 2 MHz
    
    # Multi-Channel Support (Liao et al. / ELECO 2023)
    # Channels (Hz)
    CHANNELS = {
        0: 2.4e9,
        1: 5.0e9,
        2: 5.8e9
    }
    
    # Frequency Dependent PA Efficiency (eta)
    # f < 2.5G: 0.50, 2.5-6G: 0.35, >6G: 0.19 (Simplified mapping for discrete channels)
    ETA_PA = {
        2.4e9: 0.50,
        5.0e9: 0.35,
        5.8e9: 0.19 # Assuming slight drop or use 0.35 if strictly <6. User said 2.5 <= f <= 6 is 0.35. 5.8 is <= 6.
    }
    # Correction based on user prompt: 2.5 <= f <= 6 GHz for eta = 0.35. 5.8e9 fits here.
    
    FC = 2.4e9    # Default Carrier frequency (Hz) - Deprecated for dynamic, but kept for init.
    ETA = 2.0     # Path loss exponent (Free space assumption)
    
    # Noise and Power
    N0_Linear = 10**(-100/10) * 1e-3 # -100 dBm noise floor
    
    # Power Constants (Liao et al.)
    P_UAV_TX_DBM = 20.0
    P_JAM_TX_DBM_MAX = 30.0
    
    P_TX_UAV = 10**(P_UAV_TX_DBM/10) * 1e-3 # 0.1 W
    # Note: P_TX_NODE is already in EnvConfig, usually lower (e.g. 0.1W or less)
    
    # Jammer Power Model (Cui et al.)
    P_CIRCUIT = 0.1 # W (Assumed base circuit power if not specified)
    SINR_THRESHOLD = 1.0 # 0 dB

    
    
    # Defaults for Equation 288
    L_P = 1024      # Packet length (bits)
    E_ACQ = 1e-5    # Data acquisition energy (J)
    P_ENC = 0.01    # Encryption power consumption (W)
    T_ENC = 1e-4    # Encryption time (s)
