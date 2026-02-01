# Configuration Refactoring

Separating high-level scenario and environment parameters from low-level physical constants.

- [x] **Create `core/env_config.py`**
    - [x] Define `EnvConfig` class.
    - [x] Add scenario parameters: `NUM_NODES`, `AREA_SIZE`.
    - [x] Add simulation stepping parameters: `MAX_STEPS`, `STEP_TIME`, `UAV_START_X`, `UAV_START_Y`, `UAV_START_Z`.
    - [x] Add dynamic parameters: `UAV_SPEED`, `UAV_RADIUS` (currently hardcoded in step).
    - [x] Add Attacker parameters: `ATTACKER_X`, `ATTACKER_Y`, `MAX_JAM_POWER`.

- [x] **Update `core/config.py`**
    - [x] Remove `NUM_NODES`, `AREA_SIZE`.
    - [x] Keep physical/system constants.

- [x] **Refactor `simulation/environment.py`**
    - [x] Import `EnvConfig`.
    - [x] Replace hardcoded values (`max_steps=100`, `dt=5.0`, `radius=200`, `speed=10.0`) with `EnvConfig` values.
    - [x] Use `EnvConfig` for area size and node count.

- [x] **Update Dependent Modules**
    - [x] `simulation/entities.py` (If it uses removed keys).
    - [x] `visualization/visualization.py` (Update `AREA_SIZE` import).
    - [x] `visualization/visualizer.py` (Update `AREA_SIZE` import).
    - [x] `main.py` (Pass both configs to Logger).
