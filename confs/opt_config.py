# confs/opt_config.py
# Centralized optimization search space parameters for Phase 1 (Model HPO) and Phase 2 (Reward Tuning).

class OptConfig:
    """
    Optuna Search Spaces configuration.
    Modify this file to adjust param boundaries, choices, and neural network architectures.
    """
    # --- Neural Network Architecture Choices (Phase 1) ---
    # Programmatically generated choices for all 1, 2, and 3-layer networks of [128, 256, 512]
    # Includes all 39 possible permutations (e.g., [128, 512], [512, 128], [128, 512, 128], etc.)
    ARCH_CHOICES = []

    # Helper generator to run during class setup
    @staticmethod
    def _generate_choices():
        options = [128, 256, 512]
        choices = []
        for o1 in options:
            choices.append(f"{o1}")
        for o1 in options:
            for o2 in options:
                choices.append(f"{o1},{o2}")
        for o1 in options:
            for o2 in options:
                for o3 in options:
                    choices.append(f"{o1},{o2},{o3}")
        return choices



    # --- Shared RL (PPO, DQN, PPO-LSTM) Search Space bounds ---
    RL_LR_MIN = 1e-5
    RL_LR_MAX = 1e-3
    RL_GAMMA_MIN = 0.85
    RL_GAMMA_MAX = 0.99

    # --- DQN-Specific ---
    DQN_TARGET_UPDATE_FREQ = [200, 500, 1000, 2000]

    # --- PPO-LSTM-Specific ---
    PPOLSTM_CELL_SIZE = [16, 32, 64, 128]
    PPOLSTM_MAX_SEQ_LEN = [5, 10, 20]


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

    # --- 30 Static Random Seeds for Robustness Evaluation ---
    EVAL_SEEDS = [
        42, 107, 245, 319, 480, 512, 631, 789, 804, 912,
        1023, 1150, 1289, 1344, 1492, 1588, 1699, 1723, 1850, 1999,
        2048, 2189, 2250, 2399, 2480, 2512, 2690, 2745, 2819, 2990
    ]


# Populate ARCH_CHOICES statically at load-time (out of class scope to maintain indentation)
OptConfig.ARCH_CHOICES = OptConfig._generate_choices()

