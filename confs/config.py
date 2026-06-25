



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
    B = 125e3     # Bandwidth (Hz) -> 125 kHz (Standard LoRa channel for EU868)
    
    # Multi-Channel Support (8 standard EU868 LoRa channels)
    # Channels (Hz)
    CHANNELS = {
        0: 868.1e6,  # 868.1 MHz (Default LoRaWAN Channel 1)
        1: 868.3e6,  # 868.3 MHz (Default LoRaWAN Channel 2)
        2: 868.5e6,  # 868.5 MHz (Default LoRaWAN Channel 3)
        3: 867.1e6,  # 867.1 MHz (Optional LoRaWAN Channel 4)
        4: 867.3e6,  # 867.3 MHz (Optional LoRaWAN Channel 5)
        5: 867.5e6,  # 867.5 MHz (Optional LoRaWAN Channel 6)
        6: 867.7e6,  # 867.7 MHz (Optional LoRaWAN Channel 7)
        7: 867.9e6   # 867.9 MHz (Optional LoRaWAN Channel 8)
    }
    
    # Frequency Dependent PA Efficiency (eta) - Uniform at 29.8% for 868 MHz band transceivers (based on SX1261 datasheet at +14 dBm)
    ETA_PA = {
        868.1e6: 0.298,
        868.3e6: 0.298,
        868.5e6: 0.298,
        867.1e6: 0.298,
        867.3e6: 0.298,
        867.5e6: 0.298,
        867.7e6: 0.298,
        867.9e6: 0.298
    }
    
    FC = 2.4e9    # Default Carrier frequency (Hz) - Deprecated for dynamic, but kept for init.
    ETA = 2.0     # Path loss exponent (Free space assumption)
    
    # Noise and Power
    N0_Linear = 10**(-117/10) * 1e-3 # -117 dBm noise floor (for 125 kHz bandwidth & 6 dB Noise Figure of SX1261)
    
    # Note: Transmission powers (P_TX_NODE, P_TX_UAV) are configured in EnvConfig (confs/env_config.py)
    # to avoid duplication and maintain consistency across the simulation.

    
    # Jammer Power Model (Cui et al.)
    P_CIRCUIT = 0.1 # W (Assumed base circuit power if not specified)
    SINR_THRESHOLD = -6.5 # 0 dB
    PERSISTENCE_THRESHOLD = 1 # Steps to wait before channel-switching (rule-based controller)

    
    
    # Defaults for Equation 288
    L_P = 1024      # Packet length (bits)
    E_ACQ = 1e-5    # Data acquisition energy (J)
    P_ENC = 0.01    # Encryption power consumption (W)
    T_ENC = 1e-4    # Encryption time (s)
