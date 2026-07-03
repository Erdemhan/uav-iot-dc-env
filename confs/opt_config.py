# confs/opt_config.py
# Centralized optimization search space parameters for Phase 1 (Model HPO) and Phase 2 (Reward Tuning).

class OptConfig:
    """
    Optuna Search Spaces configuration.
    Modify this file to adjust param boundaries, choices, and neural network architectures.
    """
    # --- Neural Network Architecture Choices (Phase 1) ---
    ARCH_CHOICES = [
        # Shallow (1 layer)
        [128],
        [256],
        [512],
        # Homogeneous 2-layer
        [128, 128],
        [256, 256],
        [512, 512],
        # Expanding 2-layer
        [128, 256],
        [256, 512],
        # Shrinking 2-layer (funnel)
        [256, 128],
        [512, 256],
        # Homogeneous 3-layer
        [128, 128, 128],
        [256, 256, 256],
        [512, 512, 512],
        # Expanding 3-layer
        [128, 256, 512],
        # Shrinking 3-layer
        [512, 256, 128],
        # Bottleneck 3-layer
        [256, 128, 256],
        [512, 256, 512],
    ]

    # --- Shared RL (PPO, DQN, PPO-LSTM) Search Space bounds ---
    RL_LR_MIN = 1e-5
    RL_LR_MAX = 1e-3
    RL_GAMMA_MIN = 0.85
    RL_GAMMA_MAX = 0.99

    # --- DQN-Specific ---
    DQN_TARGET_UPDATE_FREQ = [200, 500, 1000, 2000]

    # --- PPO-LSTM-Specific ---
    PPOLSTM_CELL_SIZE = [128, 256, 512]
    PPOLSTM_MAX_SEQ_LEN = [10, 20, 30]


    # --- QJC Search Space bounds ---
    QJC_TAU_0_MIN = 1e-5
    QJC_TAU_0_MAX = 1e-3
    QJC_GAMMA_MIN = 0.85
    QJC_GAMMA_MAX = 0.99
    QJC_TEMP_XI_MIN = 1.0
    QJC_TEMP_XI_MAX = 10.0
    QJC_MU_OFFSET_MIN = 1.0
    QJC_MU_OFFSET_MAX = 2.0

    # --- Phase 2 Reward Weight Search Space bounds ---
    REWARD_W_SUCCESS_MIN = 0.5
    REWARD_W_SUCCESS_MAX = 0.95
    REWARD_W_COST_MIN = 0.005
    REWARD_W_COST_MAX = 0.1
