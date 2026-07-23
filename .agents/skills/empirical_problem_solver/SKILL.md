---
name: empirical_problem_solver
description: Enforces formal engineering practices, empirical record verification, git history diffing, and zero-speculation root-cause analysis for any numerical mismatch or failure.
---

# Empirical Problem Solver Skill

## Objective
Prevent speculative hypotheses (e.g., guessing GPU differences, IP/network issues, OS quirks, or random seed behavior) when numerical discrepancies or runtime failures occur. Enforce a strict, formal engineering methodology based exclusively on empirical logs, artifact records, and git history analysis.

---

## The 4-Phase Formal Engineering Protocol

Whenever a numerical mismatch, unexpected training behavior, or script failure occurs, **you MUST follow these 4 phases sequentially**:

### Phase 1: Empirical Record & Log Gathering (NO GUESSWORK)
1. **STOP & DO NOT MODIFY SOURCE CODE**: Do not edit any implementation files based on intuition or assumptions.
2. **Extract Raw Data**: Read exact log files (`progress.csv`, `result.json`, `optuna_study/trial_*/result.json`, stdout logs).
3. **Build a Side-by-Side Comparison Table**:
   - Record exact numerical values (e.g., Power, JSR, Tracking %, Objective) at identical iteration steps (e.g., Iteration 5, 10, 15, 20).
   - Identify the precise iteration where divergence first appears.

### Phase 2: Git History & Delta Analysis (FORMAL AUDIT)
1. **Audit Formula & Parameter Changes**:
   - Run `git log -p -S "<symbol_or_formula>"` to trace when physics equations, hyperparams, or rewards were edited.
   - Run `git diff` against known-working commit hashes to inspect every line of code change.
2. **Isolate Code Drift**:
   - Compare data structure definitions, default parameters, and evaluation metrics between HPO scripts (`tune_models.py`) and training scripts (`train.py`).

### Phase 3: Minimal Deterministic Isolation (EMPIRICAL PROOF)
1. **Build a Micro-Verification Script**:
   - Write a short (10–20 second) scratch script in `scratch/` or run a single-trial test.
   - Run the exact algorithm state for 5–20 iterations under identical seeds and hyperparameters.
2. **Empirically Prove Parity**:
   - Verify that output numbers match reference records down to 4+ decimal places (e.g., `Power: 0.240889 W`, `Objective: -0.722667`).

### Phase 4: Targeted Fix & Permanent Verification
1. **Apply Minimal Root-Cause Fix**: Modify source code ONLY after empirical proof confirms the exact root cause.
2. **Update Tests & Documentation**: Update paper formulations, docstrings, or test assertions to reflect the verified empirical truth.
3. **Re-Run Micro Verification**: Confirm 100.000% numerical match before declaring resolution.

---

## Forbidden Anti-Patterns
- ❌ **Speculating Hardware/OS Causes**: Blaming GPU models, CPU architectures, WSL2, or IP addresses without log evidence.
- ❌ **Fudging/Rounding Data**: Modifying tables or reports without exact reproducibility.
- ❌ **Trial-and-Error Code Mutation**: Changing parameters blindly without tracing git diffs first.
