---
name: env_physics_validator
description: Validates physics formulations in core/physics.py and gym environment structures in simulation/pettingzoo_env.py.
---

# Environment & Physics Validator Skill

You have triggered the `env_physics_validator` skill because you are modifying simulation physics or PettingZoo environment logic.

## Objectives
Ensure correct implementation of UAV-IoT communication models, jamming impact, path loss calculations, and reinforcement learning env constraints.

## Guidelines & Rules

1. **Separation of Concerns**:
   - **`core/physics.py`**: Must contain stateless, pure functions doing mathematical calculations. No environment state, agent information, or file loading.
     - *Key Calculations*: Path Loss, received power, SINR (Signal-to-Interference-plus-Noise Ratio), and Shannon Capacity.
   - **`simulation/pettingzoo_env.py`**: Manages environment steps, state changes, reward calculations, and builds the Multi-Agent observation space.

2. **Communication & Jamming Rules**:
   - Free Space Path Loss (FSPL) and transmission models should follow standard physics equations.
   - Signal-to-Interference-plus-Noise Ratio (SINR) calculation:
     $$\text{SINR} = \frac{P_{sig} \cdot H_{sig}}{N_0 + \sum P_{jam} \cdot H_{jam}}$$
     *(Remember: Use plain text equivalent in chat, but LaTeX in paper drafts!)*
   - Jamming success depends on SINR dropping below the receiver threshold. Verify how thresholds are defined in `confs/env_config.py`.

3. **Multi-Agent Setup**:
   - The environment must follow PettingZoo API conventions.
   - Check that action/observation spaces are properly aligned and that evaluation loops match the training observation space formats.
