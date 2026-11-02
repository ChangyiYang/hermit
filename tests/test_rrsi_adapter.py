"""RRSI Adapter: the rrsi-adapter template and the Evolver's rounds over steps (baseline and delta, analysis,
drafting with upstream's done() contract, the critic with repair, tagging, check and smoke, one candidate evaluated
per step, upstream's selection, H_T) with a scripted proposer and scripted roles (no model, no Docker, no network)."""
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import pytest

pytest.importorskip("rrsi")

from hermit.algorithms.evolvers.rrsi_adapter import evolver as E  # noqa: E402
from hermit.algorithms.evolvers.rrsi_adapter import roles as R  # noqa: E402
from hermit.evolver.base import Access, Progress, StepContext, StepResult, Workspace  # noqa: E402
from hermit.spec import check_spec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "hermit" / "algorithms" / "templates" / "rrsi_adapter"

DOMAIN_PY = '''
from pathlib import Path
from hermit.algorithms.evolvers.rrsi_adapter.domain import AdapterDomain

class ToyDomain(AdapterDomain):
    name = "toy"
    texts = Path(__file__).resolve().parent / "texts"
    critic_patterns = [(r"SECRET-ANSWER", "an answer in the diff")]
'''


@pytest.fixture
def tmp_path():
    """A directory outside /tmp: the template check runs under bubblewrap with an empty tmpfs at /tmp."""
    base = Path.home() / ".cache" / "hermit-tests"
    base.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=base))
    yield path
    if not os.environ.get("KEEP_TEST_DIRS"):
        shutil.rmtree(path, ignore_errors=True)


def seed(tmp: Path) -> Path:
    adapter = tmp / "adapter"
    shutil.copytree(TEMPLATE, adapter, ignore=shutil.ignore_patterns("__pycache__"))
    (adapter / "ADAPTER.md").write_text("# Skill\n\nRead the task first.\n")
    return adapter


def domain_files(tmp: Path) -> str:
    texts = tmp / "dom" / "texts"
    texts.mkdir(parents=True)
    (texts / "SKILL.md").write_text("# Constitution\nNo task-specific content.\n")
    (texts / "PATTERNS.md").write_text("# Patterns\n")
    (texts / "briefs.json").write_text(json.dumps({k: f"{k} brief" for k in
                                                   ("analyst", "digester", "proposer", "critic")}))
    (tmp / "dom" / "domain.py").write_text(DOMAIN_PY)
    return str(tmp / "dom" / "domain.py") + ":ToyDomain"


# ---------- the template ----------

def test_template_root_and_seed_pass_the_check(tmp_path):
    assert check_spec(TEMPLATE, TEMPLATE, deploy=True).passed
    assert check_spec(seed(tmp_path), TEMPLATE, deploy=True).passed


def test_any_layout_is_learnable_but_hermit_is_not(tmp_path):
    adapter = seed(tmp_path)
    (adapter / "scripts").mkdir()
    (adapter / "scripts" / "look.py").write_text("print(1)\n")
    (adapter / "memory").mkdir()
    (adapter / "memory" / "NOTES.md").write_text("## lessons\n")
    assert check_spec(adapter, TEMPLATE, deploy=True).passed
    (adapter / ".hermit" / "EVOLVER.md").write_text("changed\n")
    result = check_spec(adapter, TEMPLATE, deploy=True)
    assert not result.passed and any(".hermit" in i.path for i in result.issues)


# ---------- component signals ----------

def test_adapter_diffs_are_tagged_by_upstreams_normalize():
    from rrsi.components import normalize
    from hermit.algorithms.evolvers.rrsi_adapter.domain import ADAPTER_SIGNALS
    script = "diff --git a/scripts/sum.py b/scripts/sum.py\n--- /dev/null\n+++ b/scripts/sum.py\n+print(1)\n"
    notes = "--- a/memory/NOTES.md\n+++ b/memory/NOTES.md\n+## a lesson\n"
    prose = "--- a/ADAPTER.md\n+++ b/ADAPTER.md\n+Check units before answering.\n"
    hooks = '--- a/hooks.json\n+++ b/hooks.json\n+  "Stop": [{"command": "python scripts/check.py"}]\n'
    assert normalize("client_tool", script, ADAPTER_SIGNALS) == "client_tool"
    assert normalize("skill", script, ADAPTER_SIGNALS) == "client_tool"        # no evidence: re-tagged from the diff
    assert normalize("memory", notes, ADAPTER_SIGNALS) == "memory"
    assert normalize("client_tool", prose, ADAPTER_SIGNALS) == "prompt"
    assert normalize("control_flow", hooks, ADAPTER_SIGNALS) == "control_flow"


# ---------- the Evolver ----------

def workspace(tmp: Path) -> Workspace:
    ws = Workspace(adapter=seed(tmp), learning_material=tmp / "material", task_info=tmp / "task-info",
                   history=tmp / "history", state=tmp / "state", scratch=tmp / "scratch", output=tmp / "output")
    for d in (ws.learning_material, ws.state, ws.scratch, ws.output, ws.task_info):
        d.mkdir(parents=True, exist_ok=True)
    return ws


def material(ws: Workspace, step: int, scores: dict, tokens: int = 1000, status: dict | None = None) -> None:
    """One step's learning material: rows `<task>#<k>` with the judge's score, tokens and a Codex trajectory."""
    d = ws.learning_material / f"step_{step:04d}"
    (d / "tasks").mkdir(parents=True)
    for row, score in scores.items():
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", row)
        traj = d / "trajectories" / slug
        traj.mkdir(parents=True)
        (traj / "agent-codex.jsonl").write_text(
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": f"working on {row}"}})
            + "\n" + json.dumps({"type": "turn.completed", "usage": {"input_tokens": tokens, "output_tokens": 1}})
            + "\n")
        (d / "tasks" / f"{slug}.json").write_text(json.dumps({
            "id": row, "input": {"question": f"question {row}"}, "output": {"answer": "x"},
            "judge": {"score": score, "feedback": "predicted 'x' but expected '41.5'"},
            "status": (status or {}).get(row, "ok"), "tokens": tokens, "trajectory": f"trajectories/{slug}"}))
    current = ws.learning_material / "current"
    if current.is_symlink():
        current.unlink()
    current.symlink_to(d.name)


def rows(scores: dict) -> dict:
    """{task: [trial scores]} as rows `<task>#<k>`."""
    return {f"{t}#{i + 1}": s for t, ss in scores.items() for i, s in enumerate(ss)}


def context(ws: Workspace, step: int, dom: str, tmp: Path, run_agent=None, **params) -> StepContext:
    def check_spec_tool(path: str) -> dict:
        result = check_spec(Path(path), TEMPLATE, deploy=True)
        return {"passed": result.passed, "issues": [i.__dict__ for i in result.issues]}
    tools = {"check_spec": check_spec_tool}
    if run_agent is not None:
        tools["run_agent"] = run_agent
    cfg = {"T": 2, "k": 2, "m": 2, "b_min": 1, "b_max": 3, "w": 3, "m_draft": 1, "delta": 0.05, "beta0": 0.1,
           "beta1": 44.5, "w_s": 0.0, "w_c": 15.0, "w_n": 0.5, "n_prune": 4, "repair_rounds": 2,
           "invalid_missing_frac": 0.2, "n_fail_traces": 3, "n_success_traces": 1}
    cfg.update(params.pop("rrsi", {}))
    return StepContext(workspace=ws, access=Access(), tools=tools,
                       params={"prompt": "PROPOSE.md", "domain": dom, "rrsi": cfg, "smoke": False,
                               "backend": "codex", **params},
                       progress=Progress(step, step, True), base=tmp)


class Script:
    """A scripted proposer and scripted roles. `plans[(t, variant)]` is a list of session actions: each writes
    files into the candidate and a done.json (or not). Roles answer by their system text."""

    def __init__(self, plans: dict, critic=None):
        self.plans, self.critic = plans, critic or (lambda diff: {"verdict": "accept", "reasons": [],
                                                                  "risk_notes": []})
        self.sessions, self.calls, self.paths, self.prompts = {}, [], {}, []

    def session(self, evolver, ctx):
        prompt = Path(ctx.params["prompt"]).read_text()
        self.prompts.append(prompt)
        if ctx.session:
            key, cand, done = self.paths[ctx.session]
        else:
            m = re.search(r"variant ([A-H]) of round (\d+)", prompt)
            key = (int(m.group(2)), m.group(1))
            cand = Path(re.search(r"Its adapter/ \((\S+)\) is your candidate", prompt).group(1))
            done = Path(re.search(r"write (\S+done\.json) with exactly", prompt).group(1))
        self.calls.append((key, bool(ctx.session)))
        actions = self.plans[key]
        action = actions.pop(0) if actions else {}
        for rel, text in action.get("files", {}).items():
            (cand / rel).parent.mkdir(parents=True, exist_ok=True)
            (cand / rel).write_text(text)
        if action.get("link"):
            rel, target = action["link"]
            (cand / rel).symlink_to(target)
        if "done" in action:
            done.write_text(json.dumps(action["done"]))
        sid = ctx.session or f"s-{key[0]}{key[1]}"
        self.paths[sid] = (key, cand, done)
        return StepResult("done", "proposed", session=sid)

    def generate(self, roles, prompt, system=None, json_only=False, **kw):
        if "batch analyst" in (system or ""):
            return json.dumps({"action": "report", "failure_modes": [{"mode": "wrong_form", "n_tasks": 2,
                                                                       "affected_tasks": ["q1"]}],
                               "capability_gaps": [], "success_habits": [{"habit": "reads_task", "n_tasks": 1}]})
        if "strict reviewer" in (system or ""):
            diff = prompt.split("=== DIFF ===", 1)[1]
            return json.dumps(self.critic(diff))
        raise AssertionError(f"unexpected role: {system[:80]}")


def edit(eid="C1", component="prompt", **kw):
    return {"id": eid, "component": component, "hypothesis": "h", "targets_mode": "wrong_form",
            "predicted_affected": ["q1"], "retroactive_check": "r", **kw}


def evolver(monkeypatch, script: Script):
    ev = E.RRSIAdapterEvolver()
    ev.directory = Path(E.__file__).resolve().parent
    monkeypatch.setattr(E.AgentEvolver, "step", lambda self, ctx: script.session(self, ctx))
    monkeypatch.setattr(R.CodexRoles, "generate", lambda self, prompt, system=None, **kw:
                        script.generate(self, prompt, system=system, **kw))
    monkeypatch.setattr(E.AgentEvolver, "finalize", lambda self, ctx: None)
    return ev


def state(ws: Workspace) -> dict:
    return json.loads((ws.state / E.STATE).read_text())


def test_a_run_baseline_rounds_selection_and_h_t(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    dom = domain_files(tmp_path)
    script = Script({
        (0, "A"): [{"files": {"ADAPTER.md": "# Skill\n\nAnswer with the bare number.\n"},
                    "done": {"summary": "form", "edits": [edit()]}}],
        (0, "B"): [{"files": {"scripts/units.py": "print('units')\n"},
                    "done": {"summary": "units", "edits": [edit(component="client_tool")]}}],
        (1, "A"): [{"files": {"memory/NOTES.md": "## lessons\n"},
                    "done": {"summary": "notes", "edits": [edit(component="memory")]}}],
        (1, "B"): [{"files": {"ADAPTER.md": "# Skill\n\nShorter.\n"},
                    "done": {"summary": "short", "edits": [edit()]}}],
    })
    ev = evolver(monkeypatch, script)
    base = {"q1": [0, 0], "q2": [1, 0], "q3": [1, 1], "q4": [0, 1]}

    # Step 1: H_0 evaluated (S 0.5); round 0 drafted: A and B stored; A handed back.
    material(ws, 1, rows(base))
    r = ev.step(context(ws, 1, dom, tmp_path))
    assert r.status == "done" and r.extra["next"] == "r00A", r.summary
    fr = json.loads((ws.state / "rrsi" / "frontier.json").read_text())
    assert fr["incumbent"]["cid"] == "h0" and fr["S_star"] == 0.5
    assert "Answer with the bare number" in (ws.adapter / "ADAPTER.md").read_text()
    st = state(ws)
    assert st["round"]["queue"] == ["r00B"] and st["candidates"]["r00B"]["status"] == "queued"
    directives = json.loads((ws.state / "rrsi" / "r0" / "directives.json").read_text())
    assert directives["b_t"] == 3 and directives["sigma_t"] == 0
    assert (ws.state / "rrsi" / "r0" / "analysis_report.json").is_file()
    rendered = ws.state / "rrsi" / "r0" / "analysis" / "rendered"
    assert (rendered / "q1.txt").is_file() and "GRADING" in (rendered / "q1.txt").read_text()
    assert "EDIT BUDGET" in (ws.scratch / "rrsi" / "r0A" / "context" / "round.md").read_text()

    # Step 2: A measured (S 0.75, +0.25 > delta, same tokens): B handed back.
    material(ws, 2, rows({"q1": [1, 1], "q2": [1, 0], "q3": [1, 1], "q4": [0, 1]}))
    r = ev.step(context(ws, 2, dom, tmp_path))
    assert r.extra["next"] == "r00B" and (ws.adapter / "scripts" / "units.py").is_file()

    # Step 3: B measured (S 0.5); round 0 decided: A wins; round 1 drafted from A.
    material(ws, 3, rows(base))
    r = ev.step(context(ws, 3, dom, tmp_path))
    fr = json.loads((ws.state / "rrsi" / "frontier.json").read_text())
    assert fr["incumbent"]["cid"] == "r00A" and fr["S_star"] == 0.75 and r.extra["next"] == "r01A"
    history = [json.loads(l) for l in (ws.state / "rrsi" / "history.jsonl").read_text().splitlines()]
    outcomes = {(h["t"], h["variant"]): h["outcome"] for h in history if h.get("edit_id")}
    # B is admissible too (inside the band, a new structural component: w_n * nu = 0.5 > 0) but A scored higher
    assert outcomes == {(0, "A"): "ACCEPTED", (0, "B"): "LOST"}
    assert (ws.adapter / "memory" / "NOTES.md").is_file()
    assert "bare number" in (ws.adapter / "ADAPTER.md").read_text()          # built on the incumbent A

    # Steps 4-5: round 1's candidates measured below the floor; H_2 = H_1; the run stops with H_T handed back.
    material(ws, 4, rows({"q1": [0, 0], "q2": [0, 0], "q3": [1, 1], "q4": [0, 0]}))
    r = ev.step(context(ws, 4, dom, tmp_path))
    assert r.extra["next"] == "r01B"
    material(ws, 5, rows({"q1": [0, 0], "q2": [0, 0], "q3": [1, 1], "q4": [0, 0]}))
    r = ev.step(context(ws, 5, dom, tmp_path))
    assert r.status == "stop" and r.extra["next"] == "r00A"
    decisions = json.loads((ws.state / "rrsi" / "r1" / "decisions.json").read_text())
    assert all("floor" in d["reason"] for d in decisions)
    final = ev.finalize(context(ws, 5, dom, tmp_path))
    assert "r00A" in final.summary and "bare number" in (ws.adapter / "ADAPTER.md").read_text()
    attribution = [json.loads(l) for l in (ws.state / "rrsi" / "attribution.jsonl").read_text().splitlines()]
    assert attribution[0]["predicted_hit"] == ["q1"]


def test_delta_is_calibrated_from_the_baseline_when_the_config_has_none(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    dom = domain_files(tmp_path)
    script = Script({(0, "A"): [{"files": {"ADAPTER.md": "x\n"}, "done": {"summary": "s", "edits": [edit()]}}],
                     (0, "B"): [{"files": {"ADAPTER.md": "y\n"}, "done": {"summary": "s", "edits": [edit()]}}]})
    ev = evolver(monkeypatch, script)
    material(ws, 1, rows({"q1": [0, 1], "q2": [1, 0], "q3": [1, 1], "q4": [0, 0]}))
    ev.step(context(ws, 1, dom, tmp_path, rrsi={"delta": None}))
    cal = json.loads((ws.state / "rrsi" / "calibration.json").read_text())
    assert cal["method"].startswith("bootstrap") and cal["delta"] > 0 and cal["jobs"] == ["h0"]


def test_an_invalid_done_goes_back_to_the_same_session(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    dom = domain_files(tmp_path)
    script = Script({
        (0, "A"): [{"files": {"ADAPTER.md": "x\n"},
                    "done": {"summary": "s", "edits": [edit("C1"), edit("C2"), edit("C3"), edit("C4")]}},
                   {"done": {"summary": "s", "edits": [edit("C1")]}}],
        (0, "B"): [{"done": {"summary": "s", "edits": [edit()]}}, {}, {}, {}],     # never changes a file
    })
    ev = evolver(monkeypatch, script)
    material(ws, 1, rows({"q1": [0, 1], "q2": [1, 0], "q3": [1, 1], "q4": [0, 0]}))
    r = ev.step(context(ws, 1, dom, tmp_path))
    assert r.extra["next"] == "r00A"
    assert script.calls[:2] == [((0, "A"), False), ((0, "A"), True)]
    assert "exceed the budget b_t = 3" in script.prompts[1]
    st = state(ws)
    assert st["round"]["variants"]["B"]["gate_failure"] == "no_proposal" and "r00B" not in st["candidates"]


def test_the_critic_rejects_a_leak_and_the_repair_is_screened_again(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    dom = domain_files(tmp_path)
    script = Script({
        (0, "A"): [{"files": {"ADAPTER.md": "The answer to q1 is SECRET-ANSWER.\n"},
                    "done": {"summary": "s", "edits": [edit()]}},
                   {"files": {"ADAPTER.md": "Check the units.\n"}, "done": {"summary": "s", "edits": [edit()]}}],
        (0, "B"): [{"files": {"ADAPTER.md": "SECRET-ANSWER\n"}, "done": {"summary": "s", "edits": [edit()]}},
                   {"done": {"summary": "s", "edits": [edit()]}}, {"done": {"summary": "s", "edits": [edit()]}}],
    })
    ev = evolver(monkeypatch, script)
    material(ws, 1, rows({"q1": [0, 1], "q2": [1, 0], "q3": [1, 1], "q4": [0, 0]}))
    r = ev.step(context(ws, 1, dom, tmp_path))
    st = state(ws)
    assert r.extra["next"] == "r00A" and (ws.adapter / "ADAPTER.md").read_text() == "Check the units.\n"
    assert st["round"]["variants"]["B"]["gate_failure"] == "critic_reject"
    critic = json.loads((ws.state / "rrsi" / "r0" / "A" / "critic_a0.json").read_text())
    assert critic["verdict"] == "reject" and "precheck" in critic["reasons"][0]
    assert any("REPAIR ROUND" in p for p in script.prompts)


def test_a_reserved_slot_is_enforced_on_the_diff(tmp_path, monkeypatch):
    """With the incumbent stalled, the last variant must edit a never-exercised component, judged by the diff."""
    ev = evolver(monkeypatch, Script({}))
    from rrsi.history import exploration
    explore = exploration(3, 1, {"prompt"}, 1)
    assert explore["sigma"] == 1 and "skill" in explore["untried"]
    prop, problems, final = ev._check_done(_done(tmp_path, [edit(component="prompt")]), 2, True, explore, True)
    assert prop is None and "RESERVED EXPLORATION SLOT" in problems[0] and not final
    prop, problems, final = ev._check_done(_done(tmp_path, [edit(component="skill")]), 2, True, explore, True)
    assert prop is not None and not problems
    prop, problems, final = ev._check_done(_done(tmp_path, []), 2, False, explore, False)
    assert prop is None and final                     # nothing changed and nothing declared: upstream's no_proposal


def _done(tmp: Path, edits: list) -> Path:
    path = tmp / "done.json"
    path.write_text(json.dumps({"summary": "s", "edits": edits}))
    return path


def test_rollouts_of_another_adapter_are_never_credited(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    dom = domain_files(tmp_path)
    script = Script({(0, "A"): [{"files": {"ADAPTER.md": "x\n"}, "done": {"summary": "s", "edits": [edit()]}}],
                     (0, "B"): [{"files": {"ADAPTER.md": "y\n"}, "done": {"summary": "s", "edits": [edit()]}}]})
    ev = evolver(monkeypatch, script)
    material(ws, 1, rows({"q1": [0, 1], "q2": [1, 0], "q3": [1, 1], "q4": [0, 0]}))
    ev.step(context(ws, 1, dom, tmp_path))
    (ws.adapter / "ADAPTER.md").write_text("something else ran\n")
    material(ws, 2, rows({"q1": [1, 1], "q2": [1, 1], "q3": [1, 1], "q4": [1, 1]}))
    r = ev.step(context(ws, 2, dom, tmp_path))
    st = state(ws)
    assert st["candidates"]["r00A"]["status"] == "queued" and r.extra["next"] == "r00A"
    assert (ws.adapter / "ADAPTER.md").read_text() == "x\n"


def test_too_many_missing_trials_are_measured_again_once_then_invalid(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    dom = domain_files(tmp_path)
    script = Script({(0, "A"): [{"files": {"ADAPTER.md": "x\n"}, "done": {"summary": "s", "edits": [edit()]}}],
                     (0, "B"): [{"files": {"ADAPTER.md": "y\n"}, "done": {"summary": "s", "edits": [edit()]}}]})
    ev = evolver(monkeypatch, script)
    material(ws, 1, rows({"q1": [0, 1], "q2": [1, 0], "q3": [1, 1], "q4": [0, 0]}))
    ev.step(context(ws, 1, dom, tmp_path))
    broken = rows({"q1": [1, 1], "q2": [1, 1], "q3": [1, 1], "q4": [1, 1]})
    material(ws, 2, broken, status={r: "error" for r in list(broken)[:4]})
    r = ev.step(context(ws, 2, dom, tmp_path))
    assert state(ws)["candidates"]["r00A"]["status"] == "retry" and r.extra["next"] == "r00A"
    material(ws, 3, broken, status={r: "error" for r in list(broken)[:4]})
    r = ev.step(context(ws, 3, dom, tmp_path))
    st = state(ws)
    assert st["candidates"]["r00A"]["status"] == "eval_invalid" and r.extra["next"] == "r00B"


def test_consecutive_failures_stop_the_run(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    dom = domain_files(tmp_path)
    ev = evolver(monkeypatch, Script({}))

    def broken(self, ctx):
        raise RuntimeError("the analyst is down")
    monkeypatch.setattr(E.RRSIAdapterEvolver, "_open", broken)
    material(ws, 1, rows({"q1": [0, 1], "q2": [1, 0]}))
    statuses = [ev.step(context(ws, i, dom, tmp_path)).status for i in (1, 2, 3)]
    assert statuses == ["failed", "failed", "stop"]


def test_the_roles_bridge_rebinds_upstreams_generate_only_while_bound():
    import rrsi.analyst
    import rrsi.critic
    import rrsi.digester
    original = rrsi.critic.generate
    with R.bound(lambda *a, **k: "{}"):
        assert rrsi.analyst.generate("x") == "{}" and rrsi.digester.generate("x") == "{}"
    assert rrsi.critic.generate is original


def test_smoke_failure_keeps_a_candidate_out(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    dom = domain_files(tmp_path)
    (tmp_path / "dom" / "domain.py").write_text(DOMAIN_PY + "    smoke_tasks = ['q3#1']\n")
    script = Script({(0, "A"): [{"files": {"ADAPTER.md": "x\n"}, "done": {"summary": "s", "edits": [edit()]}}],
                     (0, "B"): [{"files": {"ADAPTER.md": "y\n"}, "done": {"summary": "s", "edits": [edit()]}}]})
    ev = evolver(monkeypatch, script)
    seen = []

    def run_agent(task_ids, adapter=None):
        seen.append((task_ids, Path(adapter).name))
        status = "ok" if (Path(adapter) / "ADAPTER.md").read_text() == "x\n" else "error"
        return {"results": [{"id": t, "status": status, "output": None} for t in task_ids]}
    material(ws, 1, rows({"q1": [0, 1], "q2": [1, 0], "q3": [1, 1], "q4": [0, 0]}))
    r = ev.step(context(ws, 1, dom, tmp_path, run_agent=run_agent, smoke=True))
    st = state(ws)
    assert seen == [(["q3#1"], "checked"), (["q3#1"], "checked")]
    assert st["round"]["variants"]["B"]["gate_failure"] == "smoke_fail" and r.extra["next"] == "r00A"


def test_halted_rollouts_are_never_credited(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    dom = domain_files(tmp_path)
    (tmp_path / "dom" / "domain.py").write_text(DOMAIN_PY + """
    def halted_rows(self, material):
        return ["q1#1"] if (material / "HALTED").exists() else []
""")
    script = Script({(0, "A"): [{"files": {"ADAPTER.md": "x\n"}, "done": {"summary": "s", "edits": [edit()]}}],
                     (0, "B"): [{"files": {"ADAPTER.md": "y\n"}, "done": {"summary": "s", "edits": [edit()]}}]})
    ev = evolver(monkeypatch, script)
    material(ws, 1, rows({"q1": [0, 1], "q2": [1, 0], "q3": [1, 1], "q4": [0, 0]}))
    (ws.learning_material / "step_0001" / "HALTED").write_text("")
    r = ev.step(context(ws, 1, dom, tmp_path))
    assert r.status == "failed" and state(ws)["candidates"]["h0"]["status"] == "queued"
    assert not (ws.state / "rrsi" / "frontier.json").exists()
    material(ws, 2, rows({"q1": [0, 1], "q2": [1, 0], "q3": [1, 1], "q4": [0, 0]}))
    r = ev.step(context(ws, 2, dom, tmp_path))
    assert r.status == "done" and r.extra["next"] == "r00A"


def test_a_plan_refusal_in_a_proposer_session_is_not_a_gate_failure(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    dom = domain_files(tmp_path)
    ev = evolver(monkeypatch, Script({}))

    def refused(self, ctx):
        log = ctx.workspace.output / "agent-codex.jsonl"
        log.write_text(json.dumps({"type": "error", "message": "You've hit your usage limit."}) + "\n")
        return StepResult("failed", "codex exited with 1")
    monkeypatch.setattr(E.AgentEvolver, "step", refused)
    material(ws, 1, rows({"q1": [0, 1], "q2": [1, 0], "q3": [1, 1], "q4": [0, 0]}))
    r = ev.step(context(ws, 1, dom, tmp_path))
    st = state(ws)
    assert r.status == "failed" and "usage limit" in r.summary
    assert st["round"]["variants"] == {} and st["failures"] == 0
    assert not (ws.state / "rrsi" / "r0" / "A" / "prep.json").exists()


def test_staging_keeps_what_git_commits_and_the_diff_is_independent_of_the_candidate(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "ADAPTER.md").write_text("a\n")
    cand = tmp_path / "cand"
    shutil.copytree(base, cand)
    (cand / "ADAPTER.md").write_text("b\n")
    (cand / "scripts" / "__pycache__").mkdir(parents=True)
    (cand / "scripts" / "x.py").write_text("print(1)\n")
    (cand / "scripts" / "__pycache__" / "x.cpython-39.pyc").write_bytes(b"\x00\x01")
    (cand / ".pytest_cache").mkdir()
    (cand / ".pytest_cache" / "README.md").write_text("cache\n")
    (cand / "tools" / ".git").mkdir(parents=True)
    (cand / "tools" / ".git" / "HEAD").write_text("ref\n")
    (cand / "notes").mkdir()
    (cand / "notes" / "secret.md").write_text("ignored\n")
    (cand / ".gitignore").write_text("notes/\n")
    staged = tmp_path / "staged"
    dropped = E.stage(cand, staged)
    files = sorted(str(p.relative_to(staged)) for p in staged.rglob("*") if p.is_file())
    assert files == [".gitignore", "ADAPTER.md", "scripts/x.py"]
    assert {"notes/", ".pytest_cache/", "tools/.git/"} <= set(dropped)
    diff = E.diff_of(base, staged)
    assert "+b" in diff and "+++ b/scripts/x.py" in diff and "--- a/ADAPTER.md" in diff and "pyc" not in diff


def test_a_symbolic_link_in_a_candidate_is_sent_back_and_never_stored(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    dom = domain_files(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("the answers\n")
    script = Script({(0, "A"): [{"files": {"ADAPTER.md": "x\n"}, "done": {"summary": "s", "edits": [edit()]},
                                 "link": ("notes.md", secret)}, {}, {}, {}],
                     (0, "B"): [{"files": {"ADAPTER.md": "y\n"}, "done": {"summary": "s", "edits": [edit()]}}]})
    ev = evolver(monkeypatch, script)
    material(ws, 1, rows({"q1": [0, 1], "q2": [1, 0], "q3": [1, 1], "q4": [0, 0]}))
    r = ev.step(context(ws, 1, dom, tmp_path))
    st = state(ws)
    assert st["round"]["variants"]["A"]["gate_failure"] == "no_proposal" and r.extra["next"] == "r00B"
    assert any("symbolic links" in p for p in script.prompts)
    assert not any("the answers" in p.read_text() for p in (ws.state / "candidates").rglob("*") if p.is_file())


def test_components_the_agent_cannot_carry_are_never_offered_for_exploration(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    dom = domain_files(tmp_path)
    (tmp_path / "dom" / "domain.py").write_text(DOMAIN_PY + "    carriers = ['prompt', 'skill', 'memory']\n")
    script = Script({(0, "A"): [{"files": {"ADAPTER.md": "x\n"}, "done": {"summary": "s", "edits": [edit()]}}],
                     (0, "B"): [{"files": {"ADAPTER.md": "y\n"}, "done": {"summary": "s", "edits": [edit()]}}]})
    ev = evolver(monkeypatch, script)
    material(ws, 1, rows({"q1": [0, 1], "q2": [1, 0], "q3": [1, 1], "q4": [0, 0]}))
    ev.step(context(ws, 1, dom, tmp_path))
    d = json.loads((ws.state / "rrsi" / "r0" / "directives.json").read_text())
    assert d["explore"]["untried"] == ["prompt", "skill", "memory"]
    assert "subagent" in d["uncarried"] and "control_flow" in d["uncarried"]
    assert "How the diff evidences each component" in (ws.scratch / "rrsi" / "r0A" / "context" / "round.md").read_text()


def test_a_working_copy_that_never_matches_is_invalid_after_one_more_hand_back(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    dom = domain_files(tmp_path)
    script = Script({(0, "A"): [{"files": {"ADAPTER.md": "x\n"}, "done": {"summary": "s", "edits": [edit()]}}],
                     (0, "B"): [{"files": {"ADAPTER.md": "y\n"}, "done": {"summary": "s", "edits": [edit()]}}]})
    ev = evolver(monkeypatch, script)
    material(ws, 1, rows({"q1": [0, 1], "q2": [1, 0], "q3": [1, 1], "q4": [0, 0]}))
    ev.step(context(ws, 1, dom, tmp_path))
    for step in (2, 3):
        (ws.adapter / "ADAPTER.md").write_text("what actually ran\n")
        material(ws, step, rows({"q1": [1, 1], "q2": [1, 1], "q3": [1, 1], "q4": [1, 1]}))
        r = ev.step(context(ws, step, dom, tmp_path))
    st = state(ws)
    assert st["candidates"]["r00A"]["status"] == "eval_invalid" and r.extra["next"] == "r00B"
    assert st["round"]["variants"]["A"]["gate_failure"] == "eval_invalid"


def test_a_failed_hand_back_keeps_the_candidate_queued(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    dom = domain_files(tmp_path)
    script = Script({(0, "A"): [{"files": {"ADAPTER.md": "x\n"}, "done": {"summary": "s", "edits": [edit()]}}],
                     (0, "B"): [{"files": {"ADAPTER.md": "y\n"}, "done": {"summary": "s", "edits": [edit()]}}]})
    ev = evolver(monkeypatch, script)
    material(ws, 1, rows({"q1": [0, 1], "q2": [1, 0], "q3": [1, 1], "q4": [0, 0]}))
    real = E.RRSIAdapterEvolver._hand_back
    calls = {"n": 0}

    def flaky(self, ctx, st, cid):
        calls["n"] += 1
        if cid == "r00A" and calls["n"] == 1:
            raise RuntimeError("the stored files of r00A changed after it was stored")
        return real(self, ctx, st, cid)
    monkeypatch.setattr(E.RRSIAdapterEvolver, "_hand_back", flaky)
    r = ev.step(context(ws, 1, dom, tmp_path))
    assert r.status == "failed" and state(ws)["round"]["queue"] == ["r00A", "r00B"]


def test_smoke_rows_with_an_errored_trial_fail_the_smoke():
    from hermit.algorithms.evolvers.rrsi_adapter.domain import AdapterDomain
    dom = AdapterDomain()
    assert dom.smoke_failure({"status": "ok", "output": {"status": "agent_error"}})
    assert dom.smoke_failure({"status": "ok", "output": {"status": "timeout"}})
    assert not dom.smoke_failure({"status": "ok", "output": {"status": "ok"}})
    assert not dom.smoke_failure({"status": "ok", "output": {"answer": "42"}})


def test_the_digesters_shell_runs_through_the_sandbox_while_bound():
    import rrsi.digester
    original = rrsi.digester.subprocess
    calls = []

    class Fake:
        def run(self, *a, **k):
            calls.append(a)
    with R.bound(lambda *a, **k: "{}", Fake()):
        rrsi.digester.subprocess.run("grep x a.txt", shell=True)
    assert calls and rrsi.digester.subprocess is original


def test_the_digesters_shell_runs_in_the_visible_copy_of_the_traces(tmp_path):
    # Upstream's digester runs its shell with cwd = the round's rendered traces, under the state the sandbox covers;
    # the command must run in the visible copy, where its relative paths name the same files.
    state = tmp_path / "state"
    rendered = state / "rrsi" / "r0" / "analysis" / "rendered"
    rendered.mkdir(parents=True)
    (rendered / "T1.txt").write_text("hidden original\n")
    copy = tmp_path / "scratch" / "rrsi_roles" / "rendered_r0"
    copy.mkdir(parents=True)
    (copy / "T1.txt").write_text("visible copy\n")
    run = R.SandboxedSubprocess(None, [state], visible=copy).run
    out = run("cat T1.txt", shell=True, cwd=rendered, capture_output=True, text=True)
    assert out.stdout == "visible copy\n"
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "T1.txt").write_text("elsewhere\n")
    assert run("cat T1.txt", shell=True, cwd=other, capture_output=True, text=True).stdout == "elsewhere\n"
    assert R.SandboxedSubprocess(None, [state]).run("cat T1.txt", shell=True, cwd=rendered, capture_output=True,
                                                    text=True).stdout == "hidden original\n"


def test_strict_hiding_refuses_a_command_it_could_not_cover(tmp_path):
    hidden = tmp_path / "state"
    hidden.mkdir()

    class Traced:
        enabled = True

        def wrap(self, command, **kw):
            return ["strace", "-f", *command], {}
    with pytest.raises(RuntimeError, match="could not hide"):
        E.StrictHiding(Traced(), [hidden]).wrap(["codex", "exec"])

    class Off:
        enabled = False

        def wrap(self, command, **kw):
            return list(command), {}
    assert E.StrictHiding(Off(), [hidden]).wrap(["codex", "exec"])[0] == ["codex", "exec"]


def test_officeqa_denylist_names_values_and_skips_years_and_round_numbers():
    import importlib.util
    spec = importlib.util.spec_from_file_location("oqa_domain", ROOT / "examples/officeqa/rrsi_adapter/domain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = [{"judge": {"feedback": f"predicted 'x' but expected '{v}'"}}
               for v in ("1987", "100", "4.73", "57,615.04", "[10102000000, 4.73]")]
    pats = module.OfficeQADomain().leak_patterns(records)
    hits = lambda line: [why for pat, why in pats if re.search(pat, line)]  # noqa: E731
    assert not hits("+ in 1987 the series") and not hits("+ at most 100 rows")
    assert hits("+ the value 57615.04") and hits("+ use 10,102,000,000 here")
    assert "'4.73'" in hits("+ ratio 4.73")[0]


def test_a_backend_failure_in_a_proposer_session_is_retried_not_recorded(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    dom = domain_files(tmp_path)
    script = Script({(0, "A"): [{"files": {"ADAPTER.md": "x\n"}, "done": {"summary": "s", "edits": [edit()]}}],
                     (0, "B"): [{"files": {"ADAPTER.md": "y\n"}, "done": {"summary": "s", "edits": [edit()]}}]})
    ev = evolver(monkeypatch, script)
    calls = {"n": 0}

    def flaky(self, ctx):
        calls["n"] += 1
        if calls["n"] == 1:                     # the first session dies on a 429 before doing anything
            (ctx.workspace.output / "agent-codex.jsonl").write_text(
                json.dumps({"type": "error", "message": "unexpected status 429 Too Many Requests"}) + "\n")
            return StepResult("failed", "codex exited with 1")
        return script.session(self, ctx)
    monkeypatch.setattr(E.AgentEvolver, "step", flaky)
    material(ws, 1, rows({"q1": [0, 1], "q2": [1, 0], "q3": [1, 1], "q4": [0, 0]}))
    r = ev.step(context(ws, 1, dom, tmp_path, session_retry_wait_s=0))
    st = state(ws)
    assert r.status == "done" and r.extra["next"] == "r00A" and calls["n"] == 3
    assert st["round"]["variants"]["A"].get("gate_failure") is None
    assert "429" in (ws.output / "r0A" / "session_1.jsonl").read_text()       # the failed attempt's log is kept


def test_a_repair_that_changes_no_file_is_sent_back(tmp_path, monkeypatch):
    """Upstream starts each repair as a new propose() call with no edits counted: a repair that declares edits but
    writes nothing is bounced, even though the candidate differs from H_t."""
    ws = workspace(tmp_path)
    dom = domain_files(tmp_path)
    script = Script({
        (0, "A"): [{"files": {"ADAPTER.md": "SECRET-ANSWER\n"}, "done": {"summary": "s", "edits": [edit()]}},
                   {"done": {"summary": "s", "edits": [edit()]}},                       # the repair: no file change
                   {"files": {"ADAPTER.md": "Check the units.\n"}, "done": {"summary": "s", "edits": [edit()]}}],
        (0, "B"): [{"files": {"ADAPTER.md": "y\n"}, "done": {"summary": "s", "edits": [edit()]}}]})
    ev = evolver(monkeypatch, script)
    material(ws, 1, rows({"q1": [0, 1], "q2": [1, 0], "q3": [1, 1], "q4": [0, 0]}))
    r = ev.step(context(ws, 1, dom, tmp_path))
    assert any("ZERO file changes" in p for p in script.prompts)
    assert r.extra["next"] == "r00A" and (ws.adapter / "ADAPTER.md").read_text() == "Check the units.\n"
