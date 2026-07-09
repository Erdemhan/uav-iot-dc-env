# Codebase-Memory MCP Server Guidelines

This project utilizes the `codebase-memory` MCP (Model Context Protocol) server to construct and query a semantic knowledge graph of the codebase. By leveraging language server protocol (LSP) analysis and vector embeddings, the server maps symbols (classes, functions, variables) and their dependencies, enabling advanced codebase navigation and intelligence.

## Workspace & Project Identity
- **Project Root Path:** `C:/Users/Erdemhan/Desktop/OneDrive - erciyes.edu.tr/okul_msi/Projeler/DR TEZ/uav-iot-dc-env`
- **Project Index Name:** `C-Users-Erdemhan-Desktop-OneDrive-erciyes.edu.tr-okul_msi-Projeler-DR-TEZ-uav-iot-dc-env`

---

## Detailed Tool Overview & Usage Patterns

### 1. Repository Indexing & Status
*   **`list_projects()`**: Retrieves all currently indexed projects.
*   **`index_status(project: string)`**: Retrieves the indexing state of a given project.
*   **`index_repository(repo_path: string, mode?: string, persistence?: boolean)`**:
    - **Use Case:** Re-index the codebase after adding new files, refactoring, or changing function signatures.
    - **Mode Recommendation:** Use `moderate` or `full` to update semantic relationship edges and node weights.
    - **Persistence:** Set to `true` to store the compressed knowledge graph inside `.codebase-memory/graph.db.zst`.

### 2. Code Search & Snippet Retrieval
*   **`search_code(query: string, project?: string, limit?: number)`**:
    - Performs regex or keyword-based lexical search across the codebase.
*   **`get_code_snippet(symbol: string, project?: string)`**:
    - Retrieves the exact definition block (class, function, method) of a symbol. Useful to avoid reading full file contents when only one function's signature and body are needed.

### 3. Graph Architecture & Semantic Queries
*   **`get_architecture(project?: string, aspects?: string[])`**:
    - Retrieves a high-level structural summary of the project architecture.
*   **`search_graph(query: string, project?: string)`**:
    - Finds nodes (files, classes, functions) using natural language/semantic queries (e.g., "UAV mobility model", "jamming power calculation").
*   **`query_graph(query: string, project?: string)`**:
    - Executes Cypher-like queries on the codebase graph database to find specific relationship patterns.
*   **`trace_path(start_symbol: string, end_symbol: string, project?: string)`**:
    - Computes and returns the dependency/call graph path between two symbols. Crucial for tracing how changes in physical equations (e.g., in `core/physics.py`) propagate to the simulation environment (e.g., in `simulation/pettingzoo_env.py`).

---

## Mandated Agent Workflow

When you start a session or need to research a task:

1.  **Check Index Status:**
    - Call `list_projects` first.
    - Confirm if `C-Users-Erdemhan-Desktop-OneDrive-erciyes.edu.tr-okul_msi-Projeler-DR-TEZ-uav-iot-dc-env` is present.
    - If missing, run `index_repository`.
2.  **Semantic Navigation over Brute Force Reading:**
    - Do not read entire raw files using `view_file` to locate symbol definitions.
    - Instead, use `search_graph` or `search_code` to locate the files and `get_code_snippet` to view specific classes/methods.
3.  **Trace Code Impact:**
    - Before modifying any shared module (such as configuration variables, reward calculations, or physics equations), use `trace_path` or `query_graph` to list all classes and functions that import or call the module.
4.  **Refresh on Changes:**
    - If you add a new module or script (such as a new training evaluator or network architecture), run `index_repository` to integrate the new code into the knowledge graph.
