# OfficeQA (SkillOpt paper setting)

Questions about U.S. Treasury Bulletins (1939-2025). The agent gets the question with the parsed pages its answer
comes from, can read and search the question's source files and the whole corpus (under `docs/`, read-only), and
must answer inside `<answer>...</answer>`. Scoring is exact match after normalisation (lower case; commas and
punctuation other than the numeric characters and `%` removed; the words million(s), billion(s), dollar(s) and
nominal removed), no tolerance.

The agent is Codex in a container of its own, working in a directory of its own: the adapter's `ADAPTER.md` is its
instructions there (as `AGENTS.md`), and every other file of the adapter is placed in that directory at the same
relative path, so a helper script or a notes file reaches the agent only when `ADAPTER.md` tells it what is there
and when to use it; a skill under `.agents/skills/<name>/SKILL.md` is also listed to the agent by Codex's own skill
discovery in that directory. The session's commands have no network. No roles are registered and no hooks run from
these files: a sub-agent or a hook is not a lever here, so the components carried by them (subagent, control_flow,
output_plumbing, context_mgmt) are not asked for. The names `AGENTS.md`, `task.md`, `docs` and `.last_message.txt` at
the adapter's root are taken by the session itself, and a `.codex` directory is not allowed. Each question runs in a
fresh directory: nothing the agent writes during one question reaches another.
