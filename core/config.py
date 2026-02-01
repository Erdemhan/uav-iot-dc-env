



class UAVConfig:
    """
    Thesis proposal Table 3 and default simulation parameters.
    """
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
    FC = 2.4e9    # Carrier frequency (Hz) -> 2.4 GHz
    ETA = 2.0     # Path loss exponent (Free space assumption)
    
    # Noise and Power
    N0_Linear = 10**(-100/10) * 1e-3 # Noise in Watts (example -100 dBm) 
    # Note: Since the user specified N0 + I_jam, we keep N0 as linear Watts.
    
    P_TX_NODE = 0.1 # Node transmission power (Watt)
    P_TX_UAV = 0.5  # UAV transmission power (Watt)

    # IoT and Energy Parameters
    # NUM_NODES and AREA_SIZE moved to EnvConfig

    
    # Defaults for Equation 288
    L_P = 1024      # Packet length (bits)
    E_ACQ = 1e-5    # Data acquisition energy (J)
    P_ENC = 0.01    # Encryption power consumption (W)
    T_ENC = 1e-4    # Encryption time (s)
