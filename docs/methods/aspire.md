# ASPIRE

ASPIRE ([Agentic Skills Discovery for Robotics, arXiv:2607.00272](https://arxiv.org/abs/2607.00272),
[code](https://github.com/NVlabs/ASPIRE)) learns robot skills as text: a coordinator dispatches one coding agent
(an actor) per robot task; the actor writes a Code-as-Policy program, runs it in a robot execution engine that
records every primitive call with RGB keyframes, debugs it on development seeds, and reports what it learned. The
coordinator turns those reports into a shared skill library, admitting an entry only after a fresh verifier
re-solved the task from the library and the entry alone. In this port the acting agent is the example's own actor
loop on GLM 5.3 Flash, not Claude Code or Codex, and the framework's learner changes only the skill library: the
actor's harness and its instructions are fixed.

## Setup

```yaml
template: aspire                              # the skill library: skills/README.md and skills/<topic>.md
data: {train: all.txt, val: verify.txt, test: rerun.txt, step_size: 1, shuffle: false}
agent: {name: ../utils/agent/agent.py:AspireActor}
evolver: {name: aspire}
llm: {model: fireworks/accounts/fireworks/models/glm-5p3-flash, extra_body: {reasoning_effort: medium}}
```

The template is ASPIRE's library layout: `skills/README.md` (the index) and topic files `skills/<topic>.md` of
entries `## <title>`, each with **Failure signature.**, **When to apply.**, a repair (prose and, when useful, one
Python snippet that runs under the CaP program rules) and **Provenance.** The acting agent receives `ADAPTER.md`
and below it the whole library, rendered by the template's context tool (files as context: the paper has no
retrieval ranking). The template's check parses every snippet under the program sandbox's rules. No existing
template fits: `prompt-only` is one file; `continual-harness` lays skills out for Codex and carries a policy,
sub-agents and memory ASPIRE does not have; `reharness` retrieves per instance from an embedded bank.

The verifier needs its own task rows (`data.val`): the same suite with the verifier's seeds and attempt budget
(`examples/aspire_robosuite/utils/prepare.py` writes them). Only the Evolver runs them, through `run_agent`.

## The step

One step is one finished task.

```{mermaid}
flowchart TD
    R[the step's rollout: the actor's findings and skill report] --> C[coordinator: one call of the run's llm]
    L[the library] --> C
    C -->|repairs on graded entries| S1[pre-screen: op, fields, sandbox-valid snippets, privileged-access pattern]
    C -->|proposals| S2[pre-screen]
    S1 --> V
    S2 --> V[verifier: run_agent on verify-&lt;task&gt; with library + repairs + proposals]
    V -->|score >= verify_threshold| A[admit the proposals]
    V -->|below| X[proposals stay out]
    S1 --> U[apply repairs and admitted proposals]
    A --> U
    U --> K{check_spec}
    K -->|passes| N[the new library]
    K -->|fails| B[each operation alone; the failing ones left out]
```

1. The Trainer runs the actor on the step's task: explore once (ASPIRE's scene-snapshot program), the fix loop on
   the development seeds (at most `max_versions` programs, each run on every development seed, the failing seeds'
   per-primitive traces and keyframes shown to the model for the next version), optionally the evolutionary search
   (`search: evo`: K candidates per round from the top three programs and their residual failures), one run of the
   selected program on the held-out seeds, and the findings (ASPIRE's `findings.md` and `skill_report.json`).
2. The coordinator, one call of the run's `llm` with ASPIRE's promotion rules, reads the library and the task's
   report and answers with `repairs` (operations on existing entries the actor graded `wrong`, `misleading` or
   `useful`) and `proposals` (new or replaced entries from the actor's proposals).
3. Every operation is pre-screened: a repair only replaces or deletes an entry that exists, a proposal only adds
   or replaces one; the entry is one `## ` heading with its fields at the start of a line; every fenced snippet
   parses and passes the program sandbox's rules; and neither its code nor its prose names privileged access
   (simulator internals, success predicates, reflection), by a pattern, not an audit.
4. The repairs and the proposals are written into a copy of the library and the task's verifier row is run on it
   with `run_agent`: a fresh actor, which never sees the proposer's program, re-solves the task in at most two attempts on
   five seeds. The proposals are admitted when its score reaches `verify_threshold` (ASPIRE: 4 of 5).
5. The repairs and the admitted proposals (their Provenance line now carrying the verifier's result) are applied;
   a new topic file is created only below the file cap. The adapter must pass `check_spec`; when the whole update
   fails it, each operation is tried alone and the failing ones are left out.

The final adapter is the library after the last task. ASPIRE's backward transfer, one rerun of the tasks with the
final library, is the evaluation on the test split (`rerun-<task>` rows, the same seeds), compared with each task's
first pass in training (`state/aspire.json` keeps the first-pass scores; the rerun's are in the eval's results).

## Parameters

```{literalinclude} ../../hermit/agent/evolver/methods/aspire/evolver.yaml
:language: yaml
```

The actor's own settings are the example's (`agent.params`): `debug_seeds`, `eval_seeds`, `max_versions`,
`search`, `evo: {k, rounds, threshold, qualify_below, parents}`, `workers`.

## The authors' implementation against this one

| Dimension | ASPIRE (NVlabs/ASPIRE) | This port | Why |
|---|---|---|---|
| When learning happens | after each task's actor returns; the coordinator picks the next task (curriculum) | after each task (one per step), in a fixed order (`splits/all.txt`) | the Trainer's step; the curriculum's choice is a decision, not ported |
| What the learner sees | `findings.md`, `skill_report.json` | the same, from the rollout's record, and the task's development and held-out pass counts | the counts are the record's; the coordinator prompt names them |
| What the actor sees | the library, the task, its own runs; Claude Code with a shell reads any artifact | the library, the task, the scene snapshot, each version's per-seed digest of `trace/calls` and up to 2 keyframes per failed seed | the actor is a loop of `direct` calls, not a coding-agent CLI (a decision: cost and control) |
| Each component's carrier | Markdown skill files, edited by the coordinator | the same, as named-entry operations the Evolver applies | the framework applies and checks updates |
| The update | promotion after a verified reproduction; repairs on grades | the same; the verifier is the actor on a verifier row | — |
| Verification | a fresh agent, library + proposals only, 5 seeds (the template's 51-55), 2 attempts, `verified` at ≥ 4/5 | the same budget on 101-105, a subset of the development seeds; one verifier for all of a task's proposals, with the repairs | a decision: Robosuite's seed ranges; ASPIRE notes the template's seeds must be changed by hand for Robosuite |
| Pre-screen | `lint_skills.py` (sandbox validity, undefined helpers, searches without an empty case), the coordinator's API audit | sandbox validity and a privileged-access pattern; undefined helpers and empty cases are not checked | the upstream linter imports ASPIRE; the rest is left to the verifier |
| Evolutionary search | K = 8, top-3 parents, stop at 80% or 5 iterations, tasks below 90% | the same defaults | — |
| Seeds | Robosuite: development 101-125, held-out 1-100 | the same in `run.yaml` | — |
| Models | Claude Opus 4.6 (and others) acting; the coordinator the same agent | GLM 5.3 Flash acting and coordinating | the first-evaluation row of the framework |
| Perception | SAM3 | SAM3, or any service with its wire format (`utils/seg_service/`: Grounding-DINO + SAM) | SAM3's weights are gated |
| Backward transfer | rerun of tasks below 80% whose library changed | the test split reruns every task with the final library | the evaluation measures it per task |
| Fix loop | 3 versions per failure mode, 8 in total; runs every version | 8 versions in total, the model judges the modes; stops when every development seed passes | a decision: nothing is left to fix |
| Not ported | the curriculum's dynamic choice; the unseen-check seeds (126-130) and the findings' "Development vs unseen" section; LIBERO-Pro and BEHAVIOR-1K; the real robot | | the held-out run stands in for the unseen check |

## What it is not

The acting agent is not a coding-agent CLI: it does not open run artifacts itself, it is shown a digest of them.
The search over programs happens inside a task's rollout (the actor's), not in the Evolver; the Evolver learns only
the library. A verified entry transmits (a fresh actor followed it to the fix on the task it came from); it is not
shown to generalize.

## Example

[`examples/aspire_robosuite/`](https://github.com/Infini-AI-Lab/hermit/tree/main/examples/aspire_robosuite): the
seven Robosuite tasks of ASPIRE's harness.
