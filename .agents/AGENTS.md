# Customization Rules

## Response Formatting Guidelines

- **No Unrequested LaTeX in Chats:** Do not use LaTeX notation (such as `$ ... $` or `$$ ... $$`) for formulas, numbers, equations, units, or parameters in chat conversations unless the user explicitly requests it.
- **Alternative Formatting:** For chat responses, use plain text, inline code backticks, or standard markdown layout for mathematical equations and symbols (e.g., write "10^(-117/10)" or "dBm" or "W" instead of math symbols).
- **Paper/Thesis Exemption:** This rule only applies to chat conversations. You must still use standard LaTeX notation inside scientific markdown paper files (such as `paper/method_materials.md`) as required for academic publications.

## MCP Server Guidelines

- **Active codebase-memory usage:** The codebase-memory MCP server is integrated. Always ensure the project is indexed and use graph/codebase memory tools where appropriate. See [.agents/mcp.md](file:///c:/Users/Erdemhan/Desktop/OneDrive%20-%20erciyes.edu.tr/okul_msi/Projeler/DR%20TEZ/uav-iot-dc-env/.agents/mcp.md) for detailed instructions.

## Testing and Code Verification Guidelines

- **Max 50 Iterations for Testing:** When writing/running scratch scripts, debug scripts, or test command runs to verify code changes, always limit the run length to a maximum of 50 iterations (or equivalent episodes/steps) to prevent unnecessary delays.

## Debugging and Problem Solving Guidelines

- **No Guesswork on Failures:** When a failure or unexpected behavior occurs (e.g., server connection issues, empty data grids, path resolution errors), never guess the cause or propose speculative solutions. Always add comprehensive debug logs, verify exact output states, test raw endpoints locally, and identify the root cause before implementing any code changes.

