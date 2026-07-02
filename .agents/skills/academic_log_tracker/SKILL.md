---
name: academic_log_tracker
description: Guides the update of RAPOR.md in Turkish and README.md/todo.md in English after code changes.
---

# Academic Log Tracker Skill

You have triggered the `academic_log_tracker` skill because a code change or documentation change is planned or completed. 

## Objectives
Ensure all development activities are properly recorded in accordance with the project's language policy and development guidelines.

## Guidelines & Rules

1. **Language Distinction**:
   - **Code, comments, README.md, and todo.md** MUST be in **English**.
   - **RAPOR.md** MUST remain in **Turkish** (academic/thesis requirement).

2. **Gelişim Günlüğü (Section 6 of RAPOR.md)**:
   - After completing any code changes, you must append a new entry to Section 6 of `RAPOR.md`.
   - The entries must be in chronological order (oldest to newest).
   - Use the exact format:
     ```markdown
     ### [DD.MM.YYYY HH:MM] - Descriptive Title in Turkish (vX.Y.Z)
     
     **Değişiklikler:**
     - Detaylı açıklama 1 (ne değişti, neden değişti).
     - Detaylı açıklama 2.
     
     **Teknik Etki / Sonuçlar:**
     - Teknik etki açıklaması (hangi dosyaları etkiledi, çalışma performansı, vb.).
     ```
   - Make sure to check the current date/time to format the timestamp correctly.

3. **README.md and todo.md Updates**:
   - If a code change modifies the simulation loop, configs, parameters, or introduces new scripts/commands, immediately update `README.md` (in English) under the appropriate sections.
   - If a task from `todo.md` is completed, mark it off in `todo.md` (in English).
