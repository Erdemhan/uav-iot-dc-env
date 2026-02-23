
class EnvConfig:
    """
    Simulation Environment and Scenario Parameters.
    Separated from physical/system parameters.
    """
    # Scenario
    NUM_NODES = 5
    AREA_SIZE = 1000.0 # m
    
    # Attacker
    ATTACKER_POS_X = 600.0 # Absolute or relative logic can be used in env
    ATTACKER_POS_Y = 600.0 
    MAX_JAMMING_POWER = 1.0 # Watts (Industrial Standard: 30 dBm)

    # Transmission Power (Scenario specific)
    P_TX_NODE = 0.1  # Watt (20 dBm - Standard IoT/Wi-Fi)
    P_TX_UAV = 0.2   # Watt (23 dBm - Standard Telemetry)

    # Simulation Stepping
    MAX_STEPS = 100
    STEP_TIME = 5.0 # Seconds (dt)

    # Initial UAV State & Dynamics
    UAV_START_X = 500.0 # Center
    UAV_START_Y = 500.0 # Center
    UAV_START_Z = 100.0 # H (This duplicates UAVConfig.H slightly but is init pos)
    
    UAV_START_Z = 100.0 # H (This duplicates UAVConfig.H slightly but is init pos)
    
    UAV_SPEED = 5.0 # m/s (Increased for larger area)
    
    # Note: UAVConfig.H is physical operating altitude. UAV_START_Z should match or be set, 
    # but strictly speaking H is a system parameter. We can reference UAVConfig.H in env.
