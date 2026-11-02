"""ASPIRE: the aspire template (the skill library, its reader and its check), the coordinator's step with a scripted
model and a stand-in verifier (repairs, the pre-screen, the verifier gate, the file cap, the rollback), and the
example actor's pure helpers (no simulator, no model, no network)."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import pytest

from fakes import agent_of
from hermit.agent.evolver.base import Access, Progress, StepContext, Workspace
from hermit.agent.evolver.methods.aspire import evolver as E
from hermit.spec import check_spec

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "hermit" / "adapter" / "templates" / "aspire"
EXAMPLE = ROOT / "examples" / "aspire_robosuite" / "utils"

ENTRY = """## Re-localize after the approach
**Failure signature.** The gripper closes on air: the cube moved when the arm passed over it.
**When to apply.** `close_gripper()` reports a gripper width at the fully-closed limit.
Localize again from the eye-in-hand camera before the final descent.

```python
found = localize_object("red cube", camera_name="robot0_eye_in_hand", target_frame=get_robot_state().base_frame)
if not found.ok:
    raise ValueError("cube not found")
target = found.geometry.pose.position
```
**Provenance.** robosuite/cube_lifting: 1/3 -> 3/3 development seeds; verifier 5/5.
"""


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
    return adapter


def spec(path) -> dict:
    r = check_spec(Path(path), TEMPLATE, deploy=False)
    return {"passed": r.passed, "issues": [{"path": i.path, "problem": i.problem} for i in r.issues]}


def material(tmp: Path, report: dict, status: str = "ok") -> Path:
    root = tmp / "material"
    step = root / "step_0001"
    (step / "tasks").mkdir(parents=True)
    record = {"id": "robosuite-cube_lifting", "input": {"id": "robosuite-cube_lifting", "suite": "cube_lifting"},
              "output": 0.5, "judge": {"score": 0.5}, "status": status,
              "pipeline_extra": {"suite": "cube_lifting", "debug_passed": 2, "debug_total": 3, "eval_passed": 1,
                                 "eval_total": 2, "findings": "# Findings\n## Root causes observed\n- grasp missed",
                                 "skill_report": report}}
    (step / "tasks" / "robosuite-cube_lifting.json").write_text(json.dumps(record))
    (root / "current").symlink_to("step_0001")
    return root


def ctx_of(tmp: Path, adapter: Path, reply: dict, verify_score=1.0, params=None):
    calls = []

    def run_agent(task_ids, adapter=None):
        calls.append((task_ids, (Path(adapter) / "skills" / "grasp.md").read_text()))
        return {"results": [{"id": task_ids[0], "status": "ok", "judge": {"score": verify_score, "feedback": "x"}}]}

    ws = Workspace(adapter=adapter, learning_material=material(tmp, {"proposed_new": [{}]}), task_info=tmp / "ti",
                   history=tmp / "h", state=tmp / "state", scratch=tmp / "scratch", output=tmp / "out")
    for d in (ws.state, ws.scratch, ws.output):
        d.mkdir(parents=True, exist_ok=True)
    llm = agent_of(lambda messages, **kw: {"content": "```json\n" + json.dumps(reply) + "\n```"})
    ctx = StepContext(workspace=ws, access=Access(), tools={"check_spec": spec, "run_agent": run_agent}, agent=llm,
                      params={"verify": True, "verify_task": "verify-{id}", "verify_threshold": 0.8, **(params or {})},
                      progress=Progress(1, 1, False))
    return ctx, calls


# ---------- the template ----------

def test_template_root_passes_its_own_check():
    assert check_spec(TEMPLATE, TEMPLATE).passed


def test_reader_renders_the_index_then_the_skill_files(tmp_path):
    adapter = seed(tmp_path)
    (adapter / "skills" / "grasp.md").write_text((adapter / "skills" / "grasp.md").read_text().replace(
        "(no entries yet)\n", ENTRY))
    spec_ = importlib.util.spec_from_file_location("aspire_interface", adapter / "interface.py")
    module = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(module)
    out = module.library()
    assert out["files"] == ["localize.md", "grasp.md", "transport.md", "manipulation.md"]
    assert out["entries"] == 1
    assert out["text"].index("Shared Robot Skills") < out["text"].index("Re-localize after the approach")
    assert "description:" not in out["text"]


def test_check_refuses_a_broken_entry_a_bad_snippet_and_an_unlinked_file(tmp_path):
    adapter = seed(tmp_path)
    grasp = adapter / "skills" / "grasp.md"
    grasp.write_text(grasp.read_text() + "\n## No fields\nJust prose.\n\n" +
                     ENTRY.replace("## Re-localize", "## Bad snippet").replace("target = found", "import numpy\nt = "
                                                                                                   "found"))
    (adapter / "skills" / "orphan.md").write_text("---\ndescription: x\n---\n# Orphan\n")
    problems = " ".join(i["problem"] for i in spec(adapter)["issues"])
    assert "No fields: missing" in problems
    assert "Bad snippet after the approach: the snippet would be refused" in problems
    assert "orphan.md is not linked" in problems


# ---------- the coordinator ----------

def test_verified_proposal_is_admitted_and_repair_applied(tmp_path):
    adapter = seed(tmp_path)
    reply = {"repairs": [], "proposals": [{"op": "add", "file": "grasp.md", "title": "Re-localize after the approach",
                                           "entry": ENTRY, "why": "seeds closed on air"}], "notes": "one proposal"}
    ctx, calls = ctx_of(tmp_path, adapter, reply, verify_score=1.0)
    result = E.AspireEvolver().step(ctx)
    assert result.status == "done"
    assert calls and calls[0][0] == ["verify-robosuite-cube_lifting"]
    assert "Re-localize after the approach" in calls[0][1]          # the verifier saw the proposal
    assert "Re-localize after the approach" in (adapter / "skills" / "grasp.md").read_text()
    assert spec(adapter)["passed"]
    state = json.loads((ctx.workspace.state / "aspire.json").read_text())
    assert state["steps"][0]["verifications"][0]["verified"] is True
    assert "(no entries yet)" not in (adapter / "skills" / "grasp.md").read_text()
    assert "verifier verify-robosuite-cube_lifting: x (step 1)." in (adapter / "skills" / "grasp.md").read_text()


def test_refuted_proposal_stays_out(tmp_path):
    adapter = seed(tmp_path)
    before = (adapter / "skills" / "grasp.md").read_text()
    reply = {"repairs": [], "proposals": [{"op": "add", "file": "grasp.md", "title": "T", "entry": ENTRY}]}
    ctx, _ = ctx_of(tmp_path, adapter, reply, verify_score=0.4)
    E.AspireEvolver().step(ctx)
    assert (adapter / "skills" / "grasp.md").read_text() == before
    assert not (ctx.workspace.scratch / "verify" / "robosuite-cube_lifting").exists()


def test_prescreen_drops_privileged_and_malformed_proposals_without_a_verifier(tmp_path):
    adapter = seed(tmp_path)
    cheat = ENTRY.replace("target = found.geometry.pose.position", "ok = env.sim.data.qpos")
    reply = {"repairs": [], "proposals": [{"op": "add", "file": "grasp.md", "entry": cheat},
                                          {"op": "add", "file": "grasp.md", "entry": "## No fields\nprose\n"},
                                          {"op": "add", "file": "../escape.md", "entry": ENTRY}]}
    ctx, calls = ctx_of(tmp_path, adapter, reply)
    E.AspireEvolver().step(ctx)
    assert calls == []
    dropped = [p["dropped"] for p in json.loads((ctx.workspace.state / "aspire.json").read_text())
               ["steps"][0]["proposals"]]
    assert "privileged" in dropped[0] and "missing" in dropped[1] and "not a skill file" in dropped[2]


def test_repairs_cannot_add_and_proposals_cannot_delete(tmp_path):
    adapter = seed(tmp_path)
    reply = {"repairs": [{"op": "replace", "file": "grasp.md", "title": "nope", "entry": ENTRY},
                         {"op": "replace", "file": "newtopic.md", "title": "nope", "entry": ENTRY},
                         {"op": "weird", "file": "grasp.md", "title": "x", "entry": ENTRY}],
             "proposals": [{"op": "delete", "file": "grasp.md", "title": "x"}]}
    ctx, calls = ctx_of(tmp_path, adapter, reply)
    E.AspireEvolver().step(ctx)
    step = json.loads((ctx.workspace.state / "aspire.json").read_text())["steps"][0]
    assert all(r["dropped"] for r in step["repairs"]) and all(p["dropped"] for p in step["proposals"])
    assert calls == [] and not (adapter / "skills" / "newtopic.md").exists()
    assert "Re-localize" not in (adapter / "skills" / "grasp.md").read_text()


def test_the_verifier_sees_the_repairs_with_the_proposals(tmp_path):
    adapter = seed(tmp_path)
    grasp = adapter / "skills" / "grasp.md"
    grasp.write_text(grasp.read_text().replace("(no entries yet)\n", ENTRY))
    repaired = ENTRY.replace("closes on air", "closes on air (repaired)")
    second = ENTRY.replace("## Re-localize after the approach", "## Slow lift")
    reply = {"repairs": [{"op": "replace", "file": "grasp.md", "title": "Re-localize after the approach",
                          "entry": repaired}],
             "proposals": [{"op": "add", "file": "grasp.md", "title": "Slow lift", "entry": second}]}
    ctx, calls = ctx_of(tmp_path, adapter, reply)
    E.AspireEvolver().step(ctx)
    assert "(repaired)" in calls[0][1] and "## Slow lift" in calls[0][1]


def test_unlabelled_fences_and_prose_are_screened_too(tmp_path):
    adapter = seed(tmp_path)
    bare = ENTRY.replace("```python", "```").replace("target = found", "import os\ntarget = found")
    prose = ENTRY.replace("Localize again", "Read env.sim.data to localize again")
    for entry, why in ((bare, "sandbox"), (prose, "privileged")):
        assert why in E.screen({"op": "add", "file": "grasp.md", "entry": entry}, "proposals", adapter)


def test_a_heading_inside_a_snippet_is_code(tmp_path):
    adapter = seed(tmp_path)
    commented = ENTRY.replace("if not found.ok:", "## not a heading\nif not found.ok:")
    assert E.screen({"op": "add", "file": "grasp.md", "entry": commented}, "proposals", adapter) is None
    E.apply(adapter, [{"op": "add", "file": "grasp.md", "entry": commented}])
    assert spec(adapter)["passed"]
    two = ENTRY + "\n" + ENTRY.replace("## Re-localize", "## Another")
    assert "single" in E.screen({"op": "add", "file": "grasp.md", "entry": two}, "proposals", adapter)


def test_provenance_must_open_a_line(tmp_path):
    adapter = seed(tmp_path)
    inline = ENTRY.replace("\n**Provenance.**", " **Provenance.**")
    assert "Provenance" in E.screen({"op": "add", "file": "grasp.md", "entry": inline}, "proposals", adapter)


def test_the_budget_stopping_the_verifier_propagates(tmp_path):
    from hermit.trainer.budget import BudgetExceeded
    adapter = seed(tmp_path)
    reply = {"repairs": [], "proposals": [{"op": "add", "file": "grasp.md", "entry": ENTRY}]}
    ctx, _ = ctx_of(tmp_path, adapter, reply)

    def broke(task_ids, adapter=None):
        raise BudgetExceeded("usd")
    ctx.tools["run_agent"] = broke
    with pytest.raises(BudgetExceeded):
        E.AspireEvolver().step(ctx)


def test_one_bad_operation_does_not_sink_the_others(tmp_path):
    adapter = seed(tmp_path)
    good = ENTRY.replace("## Re-localize after the approach", "## Good")
    reply = {"repairs": [], "proposals": [{"op": "add", "file": "grasp.md", "entry": good},
                                          {"op": "add", "file": "transport.md", "entry": ENTRY}]}
    ctx, _ = ctx_of(tmp_path, adapter, reply)
    real = ctx.tools["check_spec"]
    ctx.tools["check_spec"] = lambda path: ({"passed": False, "issues": []}
                                            if "Re-localize" in (Path(path) / "skills" / "transport.md").read_text()
                                            else real(path))
    E.AspireEvolver().step(ctx)
    assert "## Good" in (adapter / "skills" / "grasp.md").read_text()
    assert "Re-localize" not in (adapter / "skills" / "transport.md").read_text()


def test_repairs_apply_without_the_verifier(tmp_path):
    adapter = seed(tmp_path)
    grasp = adapter / "skills" / "grasp.md"
    grasp.write_text(grasp.read_text().replace("(no entries yet)\n", ENTRY))
    fixed = ENTRY.replace("closes on air", "closes on air (narrowed: cubes only)")
    reply = {"repairs": [{"op": "replace", "file": "grasp.md", "title": "Re-localize after the approach",
                          "entry": fixed, "why": "misleading on a nut"}], "proposals": []}
    ctx, calls = ctx_of(tmp_path, adapter, reply)
    E.AspireEvolver().step(ctx)
    assert calls == []
    text = grasp.read_text()
    assert "narrowed: cubes only" in text and text.count("## Re-localize after the approach") == 1


def test_new_topic_file_respects_the_cap_and_is_indexed(tmp_path):
    adapter = seed(tmp_path)
    reply = {"repairs": [], "proposals": [{"op": "add", "file": "bimanual.md", "entry": ENTRY}]}
    ctx, _ = ctx_of(tmp_path, adapter, reply)
    E.AspireEvolver().step(ctx)
    assert "](bimanual.md)" in (adapter / "skills" / "README.md").read_text()
    assert spec(adapter)["passed"]
    for i in range(E.MAX_FILES):
        (adapter / "skills" / f"t{i}.md").write_text("---\ndescription: x\n---\n# T\n")
    changed = E.apply(adapter, [{"op": "add", "file": "overflow.md", "entry": ENTRY}])
    assert "cap" in changed[0] and not (adapter / "skills" / "overflow.md").exists()


def test_a_failed_check_restores_the_library(tmp_path):
    adapter = seed(tmp_path)
    before = E.snapshot(adapter)
    reply = {"repairs": [], "proposals": [{"op": "add", "file": "grasp.md", "entry": ENTRY}]}
    ctx, _ = ctx_of(tmp_path, adapter, reply)
    ctx.tools["check_spec"] = lambda path: {"passed": False, "issues": [{"path": "x", "problem": "refused"}]}
    E.AspireEvolver().step(ctx)
    assert E.snapshot(adapter) == before


def test_a_failed_rollout_is_skipped(tmp_path):
    adapter = seed(tmp_path)
    ctx, calls = ctx_of(tmp_path, adapter, {"repairs": [], "proposals": []})
    record = next((ctx.workspace.learning_material / "current" / "tasks").glob("*.json"))
    data = json.loads(record.read_text())
    data["status"] = "error"
    record.write_text(json.dumps(data))
    result = E.AspireEvolver().step(ctx)
    assert result.summary.startswith("unchanged") and calls == []


# ---------- the example's actor helpers ----------

def _actor_module():
    sys.path.insert(0, str(EXAMPLE / "agent"))
    try:
        spec_ = importlib.util.spec_from_file_location("aspire_actor", EXAMPLE / "agent" / "agent.py")
        module = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE / "agent"))


def test_actor_reads_seeds_and_the_outermost_program_fence():
    A = _actor_module()
    assert A.seeds_of("101-103 7") == [101, 102, 103, 7]
    assert A.seeds_of([1, "3-4"]) == [1, 3, 4]
    reply = "diagnosis\n```python\ns = '''\n```python\nx\n```\n'''\nresult = {}\n```\ntrailing"
    assert A.program_of(reply).endswith("result = {}\n")
    assert A.program_of("no program") is None
    assert A.json_of('text ```json\n{"a": 1}\n```') == {"a": 1}


def test_sim_compact_makes_traces_readable():
    sys.path.insert(0, str(EXAMPLE / "agent"))
    try:
        import sim
    finally:
        sys.path.remove(str(EXAMPLE / "agent"))
    value = {"$type": "Pose", "position": {"$array_inline": [0.123456, 1.0, 2.0], "dtype": "float64", "shape": [3]},
             "obs": {"$type": "Observation", "cameras": {}}, "big": {"$array_inline": list(range(40)), "shape": [40]}}
    out = sim.compact(value)
    assert out == {"type": "Pose", "position": [0.1235, 1.0, 2.0], "obs": "<Observation>",
                   "big": "<array None shape [40]>"}


def test_the_failure_keyframe_is_the_failing_calls_own(tmp_path):
    sys.path.insert(0, str(EXAMPLE / "agent"))
    try:
        import sim
    finally:
        sys.path.remove(str(EXAMPLE / "agent"))
    run = tmp_path / "run"
    (run / "source").mkdir(parents=True)
    trace = [{"tool": "open_gripper", "ok": True}] + [{"tool": "move_to_pose", "ok": i != 2} for i in range(4)]
    (run / "source" / "program-result.json").write_text(json.dumps({"calls": trace}))
    names = ["reset", "before-open-gripper", "after-open-gripper"]
    for i in range(3):   # the third move raised: no after-frame of its own
        names += ["before-move-to-pose"] + (["after-move-to-pose"] if i < 2 else [])
    for n, name in enumerate(names, 1):
        d = run / "media" / "keyframes" / f"{n:06d}-{name}"
        d.mkdir(parents=True)
        (d / "agentview.png").write_bytes(b"png")
    picked = [label for label, _ in sim.keyframes(run, limit=1)]
    assert picked == [f"{names.index('before-move-to-pose', 7) + 1:06d}-before-move-to-pose"]
