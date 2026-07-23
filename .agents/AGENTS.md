# Customization Rules

## Response Formatting Guidelines

- **No Unrequested LaTeX in Chats:** Do not use LaTeX notation (such as `$ ... $` or `$$ ... $$`) for formulas, numbers, equations, units, or parameters in chat conversations unless the user explicitly requests it.
- **Alternative Formatting:** For chat responses, use plain text, inline code backticks, or standard markdown layout for mathematical equations and symbols (e.g., write "10^(-117/10)" or "dBm" or "W" instead of math symbols).
- **Paper/Thesis Exemption:** This rule only applies to chat conversations. You must still use standard LaTeX notation inside scientific markdown paper files (such as `paper/method_materials.md`) as required for academic publications.

## MCP Server Guidelines

- **Mandatory codebase-memory Integration:** The `codebase-memory` MCP server is fully configured.
- **Proactive Semantic Research:** Use `search_graph`, `query_graph`, `get_code_snippet`, and `trace_path` before reading raw files or editing shared code modules.
- **Automatic Graph Re-indexing:** Always run `index_repository` (via `call_mcp_tool`) after creating new scripts, changing class/function signatures, or updating project configurations.
- **Reference Documentation:** See [.agents/mcp.md](file:///c:/Users/Erdemhan/Desktop/OneDrive%20-%20erciyes.edu.tr/okul_msi/Projeler/DR%20TEZ/uav-iot-dc-env/.agents/mcp.md) for full tool capabilities and usage patterns.

## Testing and Code Verification Guidelines

- **Max 50 Iterations for Testing:** When writing/running scratch scripts, debug scripts, or test command runs to verify code changes, always limit the run length to a maximum of 50 iterations (or equivalent episodes/steps) to prevent unnecessary delays.

## Debugging and Problem Solving Guidelines

- **STRICT NO-GUESSWORK POLICY & LOG FIRST RULE:**
  When any error, failed process (e.g., `FAILED (code 1)`), exception, or unexpected behavior occurs:
  1. **NEVER guess, speculate, or infer the cause** of a failure without empirical log evidence.
  2. **NEVER modify any source code** before inspecting and reading the exact log file or traceback.
  3. **Mandatory First Step:** The VERY FIRST action MUST be to inspect the actual log file (e.g., `cat <run_dir>/S1-A/ppo.log`) or ask the user to share the log file output.
  4. Base every diagnosis and solution strictly on verified, un-truncated log evidence.

- **ABSOLUTE ZERO ASSUMPTION RULE:**
  1. **NEVER assume or infer** hardware types (e.g., laptop vs desktop), system setups, network topologies, or execution environments without explicit user statements or empirical verification.
  2. Base all statements strictly on verified, un-truncated facts and explicit user directives.

- **NO UNPROMPTED CODE CREATION/MUTATION RULE:**
  1. **NEVER write, create, or modify any code/files** when the user asks explanatory, diagnostic, or planning questions (e.g., "what will we test first?", "explain X").
  2. Always present the explanation or test plan verbally first, and wait for explicit user directive ("apply", "write the code", "do it") before taking any file creation or editing tool action.

## System Architecture & Environment Context

- **Distinct Environments & No OneDrive on Lab Cluster:**
  - The local development PC running this assistant (Windows OS, OneDrive path) is completely separate from the Lab Head Node and GPU Cluster PCs.
  - The Lab Head Node (`adminx@DESKTOP-3S794JR`, Linux OS) and Ray Cluster Worker PCs are standalone laboratory computers. **There is NO OneDrive installed or used on the Lab Cluster computers.**
  - Code changes are pushed to GitHub from the local development PC and pulled on the Lab Head Node by the user via git.


