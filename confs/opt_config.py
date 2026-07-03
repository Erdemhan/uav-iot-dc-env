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

    # --- PPO Search Space bounds ---
    PPO_LR_MIN = 1e-5
    PPO_LR_MAX = 1e-3
    PPO_GAMMA_MIN = 0.85
    PPO_GAMMA_MAX = 0.99

    # --- DQN Search Space bounds ---
    DQN_LR_MIN = 1e-5
    DQN_LR_MAX = 1e-3
    DQN_GAMMA_MIN = 0.85
    DQN_GAMMA_MAX = 0.99
    DQN_TARGET_UPDATE_FREQ = [200, 500, 1000, 2000]

    # --- PPO-LSTM Search Space bounds ---
    PPOLSTM_LR_MIN = 1e-5
    PPOLSTM_LR_MAX = 5e-4
    PPOLSTM_GAMMA_MIN = 0.85
    PPOLSTM_GAMMA_MAX = 0.99
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
