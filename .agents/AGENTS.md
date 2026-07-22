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

