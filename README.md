# Simulation Workflow

This document explains the working logic of the `UAV_IoT_Sim` project step by step.

## Project Structure
```
uav-iot-dc-env/
├── confs/              # Configuration Files
│   ├── config.py       # System Configuration (Physics, Delay)
│   ├── env_config.py   # Environment & Scenario Config (Nodes, Power, Speed)
├── core/               # Core modules
│   ├── physics.py      # Physics engine
│   └── logger.py       # Logging system
├── simulation/         # Simulation environment
│   ├── pettingzoo_env.py # PettingZoo ParallelEnv
│   ├── controllers.py    # Rule-based controllers (UAV)
│   └── entities.py       # Entity classes
├── visualization/      # Visualization
│   ├── visualization.py # Runtime visualization
│   └── visualizer.py   # Analysis and reporting
├── logs/               # Simulation outputs
├── main.py             # Launcher
├── README.md           # Workflow documentation
└── RAPOR.md            # Technical report (Turkish)
```

## 1. Initialization - `main.py`
When the simulation is started with `python main.py`, the following happens sequentially:

1.  **Logger Setup (`SimulationLogger`)**:
    *   A new folder with the current date/time is created under `logs/` (e.g., `EXP_20260201_200542`).
    *   Simulation parameters are read from `confs/config.py` and `confs/env_config.py`, then saved as `config.json` in this folder.
    
2.  **Environment Setup (`UAV_IoT_PZ_Env`)**:
    *   **Multi-Agent Structure**: The environment is initialized as a PettingZoo `ParallelEnv`.
    *   **Agents**:
        *   `uav_0`: Controlled via velocity commands (currently by `UAVRuleBasedController`).
        *   `jammer_0`: Controlled via power commands.
        *   `node_0`..`node_N`: Passive IoT nodes.
    *   `UAVRuleBasedController`: A controller is instantiated to manage the UAV's waypoint navigation logic.
    
3.  **Visualization (`Visualization`)**:
    *   A Matplotlib window opens, and interactive mode (`plt.ion()`) is enabled.

## 2. Simulation Loop
The simulation runs for a specified number of steps (Default: 100). In each step (`env.step(action)`), the following operations are performed:

### A. Action Selection
*   **UAV (`uav_0`)**: The `UAVRuleBasedController` calculates the necessary velocity vector (`vx`, `vy`) to reach the next waypoint.
*   **Jammer (`jammer_0`)**: A random jamming power (or RL-based action) is selected.
*   **Nodes**: Passive (No-Op).
*   All actions are passed to `env.step(actions)` dictionary.

### B. Physical Calculations (`core/physics.py`)
1.  **UAV Movement**:
    *   The UAV is moved in a circular trajectory (r=200m).
    *   New position and velocity vectors are updated.
2.  **Channel and Communication**:
    *   **UAV -> Node** and **Jammer -> Node** distances are calculated.
    *   Path losses are found using `physics.calculate_path_loss`.
    *   **SINR** (Signal/Noise+Jamming) is calculated for each node.
    *   Instantaneous **Data Rate** is found using the Shannon equation.

### C. State Updates (`simulation/entities.py`)
1.  **Connection Check**: If SINR is below a certain threshold (e.g., 1.0), the node is considered "Jammed".
2.  **Age of Information (AoI)**:
    *   If connected: AoI = 0 (Reset).
    *   If not connected: AoI += time elapsed (Stale).
3.  **Energy Consumption**:
    *   **UAV**: Aerodynamic power consumption based on flight speed is calculated.
    *   **IoT Node**: Costs for data sending and encryption are calculated.

### D. Logging (`core/logger.py`)
At the end of each step, all critical data is buffered:
*   Step No
*   UAV and Attacker Positions
*   For each Node: SINR, AoI, Energy Consumed
*   Attacker's current power

### E. Visualization (`render`)
*   Points on the map are updated.
*   **Jamming Contour**: A dynamic red contour is drawn where SINR < 0 dB, showing the real-time effective jamming zone.
*   Instantaneous step and power information is written to the screen.

## 3. Termination
When the loop ends or is stopped by the user:
1.  **Data Recording**: All data in the buffer is written to the `history.csv` file.
2.  Window is closed.
3.  The location of the log files is printed to the screen.

## 4. Analysis and Visualization - `visualization/visualizer.py`
To graph the results after the simulation ends:
```bash
python visualization/visualizer.py 
# Note: Usually called automatically by main.py
```
This command finds the latest experiment folder and generates the following graphs.

### Interpreting Results (Example)

The following graphs simplify example outputs of a scenario under attack.

#### A. Trajectory and Attack Analysis (`trajectory.png`)
![Trajectory Plot](logs/EXP_20260202_025051/trajectory.png)

*   **Blue Line**: The path followed by the UAV (Visiting nodes).
*   **Red "X"**: Position of the fixed attacker (Jammer).
*   **Colored Dots**: Successful connection points. Each node has a distinct color (e.g., Node 0 is Blue, Node 1 is Orange).
*   **Red Dot (X)**: Jamming detected (Communication lost due to attack, only shown if ALL connections are lost).
*   **Gray Dot**: Out of Range (Only shown if ALL connections are lost and no jamming).
    *   *Comment:* The concentration of red dots as the UAV approaches the attacker (top right corner) confirms that the Jammer effect increases with proximity.

#### B. Metrics Analysis (`metrics_analysis.png`)
![Metrics Analysis](logs/EXP_20260202_025051/metrics_analysis.png)

This graph consists of three panels:

1.  **Top Panel (SINR & Jamming):**
    *   **Blue Line (SINR):** Signal quality.
    *   **Red Dashed Line (Jamming Power):** Attacker's power.
    *   *Comment:* When the red line rises (attack increases), the blue line (SINR) experiences sudden drops. Points falling below 0 dB (Gray line) indicate connection loss.

2.  **Middle Panel (Age of Information - AoI):**
    *   **Green Line:** Freshness of information (Lower is better).
    *   *Comment:* A "sawtooth" pattern is seen. When the line climbs upwards (Linear increase), data is not being received (Jamming or distance). The moment the line drops to zero is the moment of successful data transfer.

3.  **Bottom Panel (Energy):**
    *   **Orange Line:** Total energy consumption of the UAV.
    *   *Comment:* Increases cumulatively over time. Changes in slope indicate speed changes (Maneuver).
