# Instructions for future Codex sessions

**Before doing anything in this project, read `.claude-memory/MEMORY.md` and follow its index to load relevant memory files.** They contain user profile, project state, architecture decisions, and workflow preferences from prior conversations.

The memory files live in the project folder (not the tool's default memory path) because the user wants them durable against tool reinstalls or API/model changes.

**When updating memory**: write directly to files under `.claude-memory/` in this repo. Do NOT write to `C:\Users\14350\.Codex\projects\f--ImageWR\memory\` — that path is deprecated for this project.

**Never commit** `.claude-memory/` to git — it contains personal notes. Already covered by `.gitignore`.
