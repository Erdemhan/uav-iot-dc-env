# Agent Protocols and Guidelines

This file defines the core documentation and maintenance rules that AI agents (or developers) must follow.

## 0. Language Policy
*   **Code & Comments:** MUST be in **English**.
*   **README.md & todo.md:** MUST be in **English**.
*   **RAPOR.md:** MUST remain in **Turkish**.
    *   *Reason:* The academic report follows specific thesis requirements.

## 1. Reporting Protocol (`RAPOR.md`)
After every technical update, code change, or parameter revision, `RAPOR.md` must be checked.

*   **Technical Updates:** If changes affecting system architecture, mathematical models, or scenario flow are made, the relevant technical sections ("2. Sistem Mimarisi", "3. Matematiksel Modeller" etc.) must be updated.
*   **Change Log:** Every significant change (Bug fix, feature addition, parameter tuning) must be added to the "5. Gelişim Günlüğü" section with a **new date/time block**.
    *   Ordering: **Ascending (Oldest to Newest)**. (e.g., v1.0.0 -> v1.1.0).
*   **Technology Stack:** If a new library or tool (e.g., PettingZoo, NumPy) is added to the project, it MUST be documented in the **"2.5. Kullanılan Altyapı ve Teknolojiler"** section of `RAPOR.md`. The description should explain *what it is* and *why/how it is used* in this project.

## 2. Workflow Documentation (`README.md`)
The file explaining the project logic is named `README.md`.

*   **Consistency:** When an update changes how the system works (e.g., adding a new script, changing visualization steps), `README.md` must be updated **without breaking the existing structure**.
*   **Content:** Initialization, Simulation Loop, Termination, and Analysis steps must always be kept up-to-date.

## 3. Task Tracking (`todo.md`)
*   Planned features and bugs to fix are tracked in `todo.md`.
*   Completed tasks should be checked off, and new requirements added.
