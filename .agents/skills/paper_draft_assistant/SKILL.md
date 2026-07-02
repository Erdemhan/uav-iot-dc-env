---
name: paper_draft_assistant
description: Assists with drafting academic papers and thesis chapters under paper/, enforcing proper LaTeX usage in documents while complying with chat formatting rules.
---

# Paper Draft Assistant Skill

You have triggered the `paper_draft_assistant` skill because you are editing or discussing paper/thesis draft materials.

## Objectives
Ensure academic paper drafts are high-quality, adhere to formatting rules, and maintain proper representation of mathematical formulations.

## Guidelines & Rules

1. **LaTeX Policy & Formatting**:
   - **Inside Document Files** (e.g., `paper/` directory markdown or `.tex` files): You MUST use standard LaTeX math notation (e.g., `$P_{jam}$`, `$\text{SINR}$`, `$\gamma = 0.9$`) for formulas, parameters, and math variables.
   - **Inside Chat Conversations**: Do NOT use LaTeX. Replace all math formulas with plain text alternatives (e.g. `P_jam`, `SINR`, `gamma = 0.9`, `10^(-117/10)`, `dBm`).

2. **Conference Paper Checklist**:
   - Ensure the baseline references (e.g., Liao et al. 2025) are cited and compared accurately.
   - Include comparison results and performance metrics (Average Jammed Nodes, SINR impact, Tracking success).
   - Document the fair comparison justifications (e.g., unified seeds, equal iterations, action space flattening).
   - Explain the reward design (especially the zero-power exploit mitigation) clearly using mathematical notation in the paper.
