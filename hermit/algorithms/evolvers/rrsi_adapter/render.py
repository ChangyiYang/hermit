"""One rollout as the text trace RRSI's analyst, digesters and proposer read (upstream: domains/*/render.py).

A rollout is the learning material's record (`tasks/<row>.json`: input, output, judge, status, error, tokens, wall
time) and its trajectory directory (`trajectories/<row>/`, the host's log copied unchanged). The acting agent here is
Codex, so the trajectory is a `codex exec --json` event stream: `agent-codex.jsonl` (the OfficeQA Codex agent), or
an agent log that holds the same events (Harbor's Codex: `agent/codex.txt`). The trace is

    === TASK <task> (trial <k>, row <row>) ===            the task as the agent got it
    [step N] AGENT / COMMAND (exit) / OUTPUT / FILES / TOOL / ERROR    one block per completed item, in order
    === RUN METADATA ===                                  status, error, tokens, wall time, steps
    === GRADING ===                                       the judge's score and feedback, last, as upstream's
                                                           VERIFIER / RUBRIC block is

With `detail=False` long outputs are cut to their head and tail, as upstream's renderers cap per-message text.
"""
from __future__ import annotations

import json
from pathlib import Path

TASK_CAP = 6000
OUTPUT_CAP = 2500
OUTPUT_CAP_DETAIL = 12000
MESSAGE_CAP = 4000
EVENT_FILES = ("agent-codex.jsonl", "codex.txt", "agent-codex.txt")


def _cut(text: str, cap: int) -> str:
    text = str(text or "")
    if len(text) <= cap:
        return text
    half = cap // 2
    return f"{text[:half]}\n[... {len(text) - cap} chars cut ...]\n{text[-half:]}"


def _events(trajectory: Path | None) -> list[dict]:
    """The Codex events of a rollout's trajectory directory (the first event file found, searched by name)."""
    if trajectory is None or not trajectory.exists():
        return []
    files = [trajectory] if trajectory.is_file() else []
    if not files:
        for name in EVENT_FILES:
            files = sorted(trajectory.rglob(name))
            if files:
                break
    out = []
    for path in files[:1]:
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def _task_text(record: dict, trajectory: Path | None) -> str:
    for name in ("task.md", "prompt.txt", "instruction.md"):
        hits = sorted(trajectory.rglob(name)) if trajectory is not None and trajectory.is_dir() else []
        if hits:
            return hits[0].read_text(errors="replace")
    value = record.get("input")
    if isinstance(value, dict):
        for key in ("question", "instruction", "prompt", "task"):
            if value.get(key):
                return str(value[key])
    return json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value


def steps(events: list[dict], detail: bool) -> list[str]:
    cap = OUTPUT_CAP_DETAIL if detail else OUTPUT_CAP
    out, n = [], 0
    for e in events:
        if e.get("type") == "error":
            out.append(f"[event] ERROR: {_cut(e.get('message'), 400)}")
            continue
        if e.get("type") == "turn.failed":
            out.append(f"[event] TURN FAILED: {_cut(json.dumps(e.get('error')), 400)}")
            continue
        if e.get("type") != "item.completed":
            continue
        item = e.get("item") or {}
        kind = item.get("type")
        n += 1
        if kind == "agent_message":
            out.append(f"[step {n}] AGENT: {_cut(item.get('text'), MESSAGE_CAP)}")
        elif kind == "reasoning":
            text = item.get("text") or item.get("summary") or ""
            if text:
                out.append(f"[step {n}] REASONING: {_cut(text, 1500)}")
        elif kind == "command_execution":
            out.append(f"[step {n}] COMMAND (exit {item.get('exit_code')}, {item.get('status')}): "
                       f"{_cut(item.get('command'), 1500)}\n[step {n}] OUTPUT:\n{_cut(item.get('aggregated_output'), cap)}")
        elif kind == "file_change":
            changes = ", ".join(f"{c.get('kind')} {c.get('path')}" for c in item.get("changes") or [])
            out.append(f"[step {n}] FILES: {changes}")
        elif kind == "mcp_tool_call":
            out.append(f"[step {n}] TOOL {item.get('server')}.{item.get('tool')} ({item.get('status')}): "
                       f"{_cut(json.dumps(item.get('arguments'), ensure_ascii=False), 1500)}\n"
                       f"[step {n}] RESULT: {_cut(json.dumps(item.get('result'), ensure_ascii=False), cap)}")
        elif kind == "web_search":
            out.append(f"[step {n}] WEB SEARCH: {_cut(item.get('query'), 300)}")
        elif kind == "todo_list":
            items = "; ".join(f"[{'x' if t.get('completed') else ' '}] {t.get('text')}" for t in item.get("items") or [])
            out.append(f"[step {n}] TODO: {_cut(items, 1500)}")
        elif kind == "error":
            out.append(f"[step {n}] ERROR: {_cut(item.get('message'), 600)}")
        else:
            out.append(f"[step {n}] {str(kind).upper()}: {_cut(json.dumps(item, ensure_ascii=False), 800)}")
    return out


def render(rec: dict, detail: bool = False) -> str:
    """The trace of one rollout. `rec`: {task_id, trial, row, record (the material record), trajectory (its
    directory or None), grading (extra lines for the GRADING block, e.g. the verifier's output)}."""
    record = rec.get("record") or {}
    trajectory = Path(rec["trajectory"]) if rec.get("trajectory") else None
    events = _events(trajectory)
    judge = record.get("judge") or {}
    usage = [e.get("usage") for e in events if e.get("type") == "turn.completed" and e.get("usage")]
    lines = [f"=== TASK {rec.get('task_id')} (trial {rec.get('trial')}, row {rec.get('row')}) ===",
             _cut(_task_text(record, trajectory), TASK_CAP), "", "=== TRAJECTORY ==="]
    body = steps(events, detail)
    lines += body or ["(no agent events were recorded)"]
    output = record.get("output")
    lines += ["", "=== RUN METADATA ===",
              f"status: {record.get('status')}" + (f" | error: {_cut(record.get('error'), 600)}"
                                                   if record.get("error") not in (None, "", "None") else ""),
              f"tokens: {record.get('tokens')} | wall_s: {record.get('wall_s')} | completed items: "
              f"{sum(1 for e in events if e.get('type') == 'item.completed')} | turns: {len(usage)}",
              f"final output: {_cut(json.dumps(output, ensure_ascii=False) if not isinstance(output, str) else output, 3000)}",
              "", "=== GRADING ===",
              f"score: {judge.get('score')}",
              f"feedback: {_cut(judge.get('feedback'), 6000 if detail else 2500)}"]
    lines += [str(x) for x in rec.get("grading") or []]
    return "\n".join(lines) + "\n"
