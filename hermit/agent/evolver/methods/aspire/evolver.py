"""AspireEvolver: the coordinator of ASPIRE (Agentic Skills Discovery for Robotics; arXiv:2607.00272, §2.2 and §2.4;
github.com/NVlabs/ASPIRE, `.claude/skills/iterative-debugging/main-agent-prompt.md` and `verifier-prompt.md`) as an
Evolver on the `aspire` template.

In ASPIRE a coordinator dispatches one actor per robot task; each actor debugs its program on development seeds and
reports `findings.md` and `skill_report.json` (grades of the library entries it consulted, proposed edits, proposed
new entries). Only the coordinator writes the shared skill library, and only through a gate: it acts on the grades
first (repairs a `wrong` or `misleading` entry, adds evidence to a `useful` one), then sends the proposals to a
verifier, a fresh actor that re-solves the task from the library and the proposals alone, and promotes them only if
the verifier reproduces the fix (at least 4 of 5 seeds within 2 attempts).

On the framework's step model one step is one finished task (`learning_material/current`): its record carries the
actor's findings and skill report (`pipeline_extra`). A step runs:

1. the coordinator, one call of the run's model: the library and the task's report in, `repairs` and `proposals`
   out (named entries of the skill files);
2. the pre-screen: each proposed entry is checked as the library's check does (the four fields, a snippet the CaP
   sandbox accepts) and for privileged access (simulator internals, reward, success predicates); a failing proposal
   is dropped;
3. the verifier: the surviving proposals are written into a copy of the library, and the task's verifier row (the
   same suite, the verifier's seeds and attempt budget; `verify_task` below) is run on it with `run_agent`; the
   proposals are admitted when its score reaches `verify_threshold`;
4. the update: the repairs and the admitted proposals are applied to the adapter, the file cap is respected (an
   entry for a new topic file is dropped at the cap; the model is told the budget), and `check_spec` must pass, or
   the adapter is restored.

Backward transfer (ASPIRE §2.4: one rerun of the low-scoring tasks with the final library) is the evaluation of the
final adapter on the tasks, compared with each task's first pass during training (`state/aspire.json` keeps both).
Training State: `state/aspire.json`; each step's `output/report.md` says what changed and why.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil

from hermit.agent.evolver.base import Evolver, ask, FinalResult, StepContext, StepResult

HERE = Path(__file__).resolve().parent
FIELDS = ("**Failure signature.**", "**When to apply.**", "**Provenance.**")   # each at the start of a line
TOPIC = re.compile(r"[a-z0-9][a-z0-9_\-]*\.md")
FILE_MAX = 24_000              # bytes, as the template's .hermit/adapter.yaml
MAX_FILES = 12                 # skill files besides the index
# Privileged access a generated program must never use (ASPIRE's provenance rule): simulator internals, the reward,
# the success predicate, reflection. The sandbox refuses `_` names already; these are the public-looking ones,
# searched in the code and in the prose of an entry alike.
PRIVILEGED = re.compile(r"\b(?:env\.sim|sim\.(?:data|model)|sim\.forward|check_success|get_reward|"
                        r"native_env|handle\.env|parsed_problem|eval_predicate|obj_body_id|getattr|setattr)\b")
FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)          # every fenced block, labelled or not
FORBIDDEN_NODES = ("AsyncFunctionDef", "Await", "ClassDef", "Delete", "Global", "ImportFrom", "Nonlocal", "With")
OPS = {"repairs": ("replace", "delete"), "proposals": ("add", "replace")}


# ---------- the library as entries ----------

def headings(text: str) -> list[int]:
    """The line numbers of the `## ` headings outside fenced blocks (a `## ` comment in a snippet is code)."""
    out, fenced = [], False
    for i, line in enumerate(text.split("\n")):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and line.startswith("## "):
            out.append(i)
    return out


def split_file(text: str) -> tuple[str, list[tuple[str, str]]]:
    """A skill file as (head, [(title, entry text from '## ' on)])."""
    lines = text.split("\n")
    marks = headings(text)
    if not marks:
        return text, []
    head = "\n".join(lines[:marks[0]])
    entries = []
    for a, b in zip(marks, marks[1:] + [len(lines)]):
        entries.append((lines[a][3:].strip(), "\n".join(lines[a:b]).rstrip() + "\n"))
    return head, entries


def join_file(head: str, entries: list[tuple[str, str]]) -> str:
    head = re.sub(r"\n*\(no entries yet\)\s*$", "\n", head.rstrip() + "\n")
    body = "\n".join(e for _, e in entries)
    return (head.rstrip() + "\n\n" + body).rstrip() + "\n" if entries else head.rstrip() + "\n\n(no entries yet)\n"


def title_of(entry: str) -> str:
    return entry.lstrip().split("\n", 1)[0][3:].strip()


def library_text(adapter: Path) -> str:
    skills = adapter / "skills"
    parts = []
    for p in [skills / "README.md", *sorted(q for q in skills.glob("*.md") if q.name != "README.md")]:
        if p.is_file():
            parts.append(f"=== skills/{p.name} ===\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


def snippet_problem(src: str) -> str | None:
    """Why the CaP sandbox would refuse a snippet (ASPIRE's _ProgramValidator rules), or None."""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return f"does not parse: {exc.msg} (line {exc.lineno})"
    for node in ast.walk(tree):
        if type(node).__name__ in FORBIDDEN_NODES:
            return f"{type(node).__name__} is not allowed in generated programs"
        if isinstance(node, ast.Import) and any(a.name != "math" or a.asname for a in node.names):
            return "only `import math` is allowed"
        if isinstance(node, (ast.Name, ast.Attribute)) and (getattr(node, "id", None) or
                                                             getattr(node, "attr", "")).startswith("_"):
            return "private and dunder names are not allowed"
    return None


def screen(op: dict, kind: str, adapter: Path) -> str | None:
    """Why an operation is dropped before it reaches the library or the verifier, or None (ASPIRE's lint_skills.py
    pre-screen and the coordinator's allowed-API check). A repair only replaces or deletes an entry that exists; a
    proposal adds or replaces one; an entry is one `## ` heading with its fields, its snippets valid programs."""
    if op.get("op") not in OPS[kind]:
        return f"a {kind[:-1]} is one of {'/'.join(OPS[kind])}, not {op.get('op')!r}"
    name = str(op.get("file") or "")
    if not TOPIC.fullmatch(name) or name == "README.md":
        return f"not a skill file name: {name!r}"
    path = adapter / "skills" / name
    titles = [t for t, _ in split_file(path.read_text(encoding="utf-8"))[1]] if path.is_file() else []
    if kind == "repairs" and str(op.get("title") or "").strip() not in titles:
        return f"a repair names an entry that exists; {name} has no {op.get('title')!r}"
    if op.get("op") == "delete":
        return None
    entry = str(op.get("entry") or "").strip() + "\n"
    if not entry.startswith("## "):
        return "the entry does not start with '## <title>'"
    if len(headings(entry)) != 1:
        return "one operation is one entry: a single '## ' heading"
    lines = entry.split("\n")
    missing = [f for f in FIELDS if not any(line.startswith(f) for line in lines)]
    if missing:
        return f"missing {', '.join(missing)} (each at the start of a line)"
    for snippet in FENCE.findall(entry):
        why = snippet_problem(snippet)
        if why:
            return f"snippet refused by the program sandbox: {why}"
    found = PRIVILEGED.search(entry)
    if found:
        return f"the entry names privileged access ({found.group(0)})"
    return None


def stamp(op: dict, verdict: str) -> dict:
    """The operation with the verifier's result on its entry's Provenance line (an entry states what is true now:
    a pending verification is replaced, not kept beside the result)."""
    entry = str(op.get("entry") or "")
    lines = entry.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("**Provenance.**"):
            line = re.sub(r"[;,.]?\s*verifier:?\s*(?:pending|not yet\s*\w*|unverified)[^;.]*", "", line,
                          flags=re.I).rstrip(" ;.")
            lines[i] = f"{line}; {verdict}."
            break
    return {**op, "entry": "\n".join(lines) + "\n"}


def apply(adapter: Path, ops: list[dict]) -> list[str]:
    """Apply screened entry operations to the library in place; what changed, as lines. `replace` puts the entry in
    place of the one titled `title` (or of the one with its own heading; otherwise it adds it); a new topic file is
    created (with a frontmatter and an index row) only below the file cap; an entry is dropped when it would push
    its file past the byte cap."""
    skills = adapter / "skills"
    changed = []
    for op in ops:
        name = op["file"]
        path = skills / name
        entry = str(op.get("entry") or "").strip() + "\n"
        new_title = title_of(entry) if entry.strip() else ""
        title = str(op.get("title") or "").strip() or new_title
        if not path.is_file():
            if op.get("op") == "delete":
                continue
            count = len([p for p in skills.glob("*.md") if p.name != "README.md"])
            if count >= MAX_FILES:
                changed.append(f"dropped {name} § {new_title}: the library is at its {MAX_FILES}-file cap")
                continue
            stem = name[:-3]
            path.write_text(f"---\nname: {stem}\ndescription: {stem.replace('-', ' ').replace('_', ' ')} "
                            f"techniques.\n---\n\n# {stem.replace('_', ' ').title()}\n\n(no entries yet)\n")
            index = skills / "README.md"
            if not index.is_file():
                index.write_text("# Shared Robot Skills\n")
            _add_index_row(index, name, stem)
        head, entries = split_file(path.read_text(encoding="utf-8"))
        titles = [t for t, _ in entries]
        if op.get("op") == "delete":
            if title in titles:
                path.write_text(join_file(head, [(t, e) for t, e in entries if t != title]), encoding="utf-8")
                changed.append(f"deleted {name} § {title}")
            continue
        target = title if title in titles else new_title if new_title in titles else None
        if target is not None:
            entries = [(new_title, entry) if t == target else (t, e) for t, e in entries]
            # a renamed entry must not collide with another of the same title
            seen, unique = set(), []
            for t, e in entries:
                if t not in seen:
                    seen.add(t)
                    unique.append((t, e))
            entries, verb = unique, "replaced"
        else:
            entries.append((new_title, entry))
            verb = "added"
        text = join_file(head, entries)
        if len(text.encode()) > FILE_MAX:
            changed.append(f"dropped {name} § {new_title}: the file would exceed {FILE_MAX} bytes")
            continue
        path.write_text(text, encoding="utf-8")
        changed.append(f"{verb} {name} § {new_title}")
    return changed


def _add_index_row(index: Path, name: str, stem: str) -> None:
    text = index.read_text(encoding="utf-8")
    if f"]({name})" in text:
        return
    row = f"| [{name}]({name}) | {stem.replace('_', ' ')} |"
    lines = text.splitlines()
    last = max((i for i, l in enumerate(lines) if l.startswith("| [")), default=None)
    if last is None:
        lines += ["", "| Skill | Purpose |", "|---|---|", row]
    else:
        lines.insert(last + 1, row)
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_reply(text: str):
    match = re.search(r"```json[^\n]*\n(.*)```", text or "", re.S)
    body = match.group(1) if match else (text or "")[(text or "").find("{"):(text or "").rfind("}") + 1]
    try:
        return json.loads(body)
    except ValueError:
        return None


def read_records(material: Path) -> list[dict]:
    current = material / "current"
    return [json.loads(p.read_text()) for p in sorted((current / "tasks").glob("*.json"))] if current.exists() else []


def snapshot(adapter: Path) -> dict[str, bytes]:
    return {p.relative_to(adapter).as_posix(): p.read_bytes() for p in (adapter / "skills").rglob("*") if p.is_file()}


def restore(adapter: Path, files: dict[str, bytes]) -> None:
    for p in list((adapter / "skills").rglob("*")):
        if p.is_file() and p.relative_to(adapter).as_posix() not in files:
            p.unlink()
    for rel, data in files.items():
        (adapter / rel).parent.mkdir(parents=True, exist_ok=True)
        (adapter / rel).write_bytes(data)


# ---------- the coordinator ----------

class AspireEvolver(Evolver):
    def step(self, ctx: StepContext) -> StepResult:
        p, ws = ctx.params, ctx.workspace
        if ctx.agent is None:
            return StepResult("failed", "aspire needs the run's model for the coordinator: set `llm` in the run config")
        adapter = ws.adapter
        if not (adapter / "skills").is_dir():
            return StepResult("failed", "the adapter is not an aspire adapter (no skills/): use the aspire template")
        n = ctx.progress.step if ctx.progress else 0
        state = self._state(ws)
        record = {"step": n, "tasks": [], "repairs": [], "proposals": [], "verifications": [], "changed": []}
        for task in read_records(ws.learning_material):
            if self._cancelled(ctx):
                return StepResult("failed", "stopped by the Trainer: this step's remaining tasks were not read")
            self._task(ctx, task, record, state)
        record["outcome"] = ("changed " + "; ".join(record["changed"])) if record["changed"] else \
            "unchanged: nothing admitted"
        if record.get("refused"):
            record["outcome"] += "; refused by check_spec: " + "; ".join(record["refused"])
        state["steps"].append(record)
        (ws.state / "aspire.json").write_text(json.dumps(state, ensure_ascii=False, indent=1))
        (ws.output / "report.md").write_text(self._report(record), encoding="utf-8")
        return StepResult("done", record["outcome"][:300], extra={k: record[k] for k in
                                                                  ("repairs", "proposals", "verifications")})

    def _task(self, ctx: StepContext, task: dict, record: dict, state: dict) -> None:
        p, adapter = ctx.params, ctx.workspace.adapter
        extra = task.get("pipeline_extra") or {}
        tid = task.get("id")
        score = (task.get("judge") or {}).get("score")
        first = state["tasks"].setdefault(tid, {"first_score": score, "first_step": record["step"]})
        first["last_score"] = score
        record["tasks"].append({"id": tid, "status": task.get("status"), "score": score,
                                "debug": f"{extra.get('debug_passed')}/{extra.get('debug_total')}"})
        report = extra.get("skill_report")
        findings = extra.get("findings") or ""
        if task.get("status") != "ok" or not (findings or report):
            return
        # 1. the coordinator
        budget = MAX_FILES - len([q for q in (adapter / "skills").glob("*.md") if q.name != "README.md"])
        prompt = (f"=== THE LIBRARY ({budget} new skill file(s) may still be created; at most {FILE_MAX} bytes per "
                  f"file) ===\n{library_text(adapter)}\n\n=== TASK {tid}: {extra.get('suite')} ===\n"
                  f"Development: {extra.get('debug_passed')}/{extra.get('debug_total')} seeds; held-out: "
                  f"{extra.get('eval_passed')}/{extra.get('eval_total')}.\n\n=== FINDINGS ===\n{findings[-30000:]}\n\n"
                  f"=== SKILL REPORT ===\n{json.dumps(report, indent=1, ensure_ascii=False)[-20000:]}")
        system = (HERE / "prompts" / "coordinator.md").read_text().split("-->", 1)[1].strip()
        sampling = {k: p[k] for k in ("max_tokens", "temperature") if p.get(k) is not None}
        reply = None
        for _ in range(3):
            answer = ask(ctx.agent, [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                         **sampling)
            reply = json_reply(answer["content"])
            if isinstance(reply, dict):
                break
        if not isinstance(reply, dict):
            record["repairs"].append({"task": tid, "error": "the coordinator gave no JSON"})
            return
        repairs = [o for o in reply.get("repairs") or [] if isinstance(o, dict)]
        proposals = [o for o in reply.get("proposals") or [] if isinstance(o, dict)]
        before = snapshot(adapter)
        # 2. the pre-screen
        kept_repairs, kept = [], []
        for kind, ops, keep in (("repairs", repairs, kept_repairs), ("proposals", proposals, kept)):
            for op in ops:
                why = screen(op, kind, adapter)
                record[kind].append({"task": tid, "file": op.get("file"), "title": op.get("title"),
                                     "op": op.get("op"), "dropped": why})
                if why is None:
                    keep.append(op)
        # 3. the verifier: the library with the repairs and the proposals, re-solved by a fresh actor on the task's
        # verifier row (ASPIRE repairs first, 6a, then verifies, 6b: what is admitted is what was verified)
        admitted: list[dict] = []
        if kept and p.get("verify", True):
            if self._cancelled(ctx):
                record["changed"].append(f"{tid}: stopped by the Trainer before the verifier; nothing applied")
                return
            verdict = self._verify(ctx, task, kept_repairs + kept)
            record["verifications"].append({"task": tid, **verdict})
            if verdict.get("verified"):
                note = f"verifier {verdict.get('task_id')}: {verdict.get('feedback') or verdict.get('score')}"
                admitted = [stamp(op, f"{note} (step {record['step']})") for op in kept]
        elif kept:
            admitted = kept
        # 4. the update, checked; when the whole update fails the check, each operation is tried alone
        ops = kept_repairs + admitted
        changed = apply(adapter, ops)
        if not self._passes(ctx, adapter):
            restore(adapter, before)
            changed = []
            for op in ops:
                one = snapshot(adapter)
                done = apply(adapter, [op])
                if self._passes(ctx, adapter):
                    changed += done
                else:
                    restore(adapter, one)
                    record["refused"] = record.get("refused", []) + [f"{tid}: {op.get('file')} § {op.get('title')}"]
        record["changed"] += [f"{tid}: {c}" for c in changed]
        if reply.get("notes"):
            record.setdefault("notes", []).append(f"{tid}: {reply['notes']}")

    @staticmethod
    def _passes(ctx, adapter: Path) -> bool:
        check = ctx.tools["check_spec"](path=str(adapter)) if "check_spec" in ctx.tools else {"passed": True}
        return bool(check.get("passed", True))

    @staticmethod
    def _cancelled(ctx) -> bool:
        return ctx.cancel is not None and ctx.cancel.is_set()

    def _verify(self, ctx: StepContext, task: dict, ops: list[dict]) -> dict:
        """ASPIRE's verifier: a fresh actor re-solves the task from the library and the proposals alone. A verifier
        row the run does not have, or an adapter run_agent refuses, leaves the proposals out (ValueError); the
        budget and the Trainer's stop propagate."""
        p = ctx.params
        vid = str(p.get("verify_task", "verify-{id}")).format(id=task["id"], suite=(task.get("input") or {})
                                                              .get("suite", ""))
        if "run_agent" not in ctx.tools:
            return {"verified": False, "reason": "no run_agent tool"}
        target = ctx.workspace.scratch / "verify" / re.sub(r"[^\w\-]", "_", str(task["id"]))
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ctx.workspace.adapter, target, symlinks=True)
        apply(target, ops)
        try:
            result = ctx.tools["run_agent"](task_ids=[vid], adapter=str(target))["results"][0]
        except ValueError as exc:
            return {"verified": False, "task_id": vid, "reason": f"{type(exc).__name__}: {exc}"[:400]}
        finally:
            shutil.rmtree(target, ignore_errors=True)
        score = (result.get("judge") or {}).get("score")
        try:
            ok = result.get("status") == "ok" and float(score) >= float(p.get("verify_threshold", 0.8))
        except (TypeError, ValueError):
            ok = False
        return {"verified": bool(ok), "task_id": vid, "score": score, "status": result.get("status"),
                "feedback": (result.get("judge") or {}).get("feedback")}

    def finalize(self, ctx) -> FinalResult:
        state = self._state(ctx.workspace)
        admitted = sum(1 for s in state["steps"] for v in s["verifications"] if v.get("verified"))
        tried = sum(len(s["verifications"]) for s in state["steps"])
        return FinalResult(adapter=ctx.workspace.adapter,
                           summary=f"aspire: {admitted} of {tried} verifications admitted proposals over "
                                   f"{len(state['steps'])} steps",
                           extra={"tasks": state["tasks"]})

    @staticmethod
    def _state(ws) -> dict:
        path = ws.state / "aspire.json"
        return json.loads(path.read_text()) if path.exists() else {"steps": [], "tasks": {}}

    @staticmethod
    def _report(record: dict) -> str:
        lines = [f"# ASPIRE coordinator, step {record['step']}", "", record["outcome"], ""]
        for t in record["tasks"]:
            lines.append(f"- task {t['id']}: status {t['status']}, score {t['score']}, development {t['debug']}")
        for key in ("repairs", "proposals"):
            for op in record[key]:
                lines.append(f"- {key[:-1]} {op.get('op')} {op.get('file')} § {op.get('title')}"
                             + (f": dropped, {op['dropped']}" if op.get("dropped") else ""))
        for v in record["verifications"]:
            lines.append(f"- verifier {v.get('task_id')}: {'verified' if v.get('verified') else 'refuted'} "
                         f"(score {v.get('score')}; {v.get('feedback') or v.get('reason') or ''})")
        for note in record.get("notes", []):
            lines.append(f"- {note}")
        return "\n".join(lines) + "\n"
