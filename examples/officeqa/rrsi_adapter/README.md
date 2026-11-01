# OfficeQA with RRSI Adapter

[RRSI](https://arxiv.org/abs/2609.24972) (Xia et al. 2026; [code](https://github.com/google-research/rrsi)),
regularized recursive self-improvement, with the harness confined to the adapter, on OfficeQA in the SkillOpt
paper's setting, with Codex as the acting agent. RRSI evolves the harness program itself; here the agent is fixed and
only the adapter changes. The [`rrsi_adapter`](../../../docs/methods/rrsi_adapter.md) Evolver runs upstream's code
for every rule and for the analyst and the critic, with the proposer and the search roles as Codex sessions; each
round's two candidates are evaluated on the 50 training questions with two trials each (the evolve set), the
incumbent after the last round (H_T) is evaluated once on the 172 test questions. The adapter is the
[`rrsi-adapter`](../../../hermit/algorithms/templates/rrsi_adapter/) template: as in Meta-Harness Adapter, every file
outside `.hermit/` is the proposer's to write.

```bash
python examples/officeqa/rrsi_adapter/make_seed.py      # ADAPTER.md = SkillOpt's initial skill (H_0)
python examples/officeqa/rrsi_adapter/make_rows.py      # rows/: each training question twice, `<id>#1`, `<id>#2`
export OFFICEQA_CODEX_IMAGE=officeqa-codex:0.164.0 OFFICEQA_CODEX_PROXY=http://oq-egress-chatgpt:3128
cd examples/officeqa/rrsi_adapter
hermit train --config smoke_luna.yaml                  # 4 questions x 2, one round of two: a check of the plumbing
hermit train --config run_luna.yaml                    # 10 rounds of two
hermit eval  --config run_luna.yaml --adapter final    # H_T once on the 172 test questions
```

The setting is the README's GPT-5.6-Luna column ([the recipe](../../../docs/recipes/officeqa.md#the-flash-row-with-gpt-56-luna);
the image, the egress proxy and the login: [A model on a ChatGPT login](../../../docs/recipes/officeqa.md#a-model-on-a-chatgpt-login)):
Codex 0.164.0 with GPT-5.6-Luna at medium effort acts in a container per question on the machine's ChatGPT login,
with Meta-Harness Adapter's deployment (`deploy: adapter`: `ADAPTER.md` as `AGENTS.md`, every other file of the
adapter in the session's directory) and 3,600 s per session; the proposer and the search roles are Codex sessions with
the same model on the same login in hermit's sandbox, which reach chatgpt.com only. No API key is used.

| File | What it is |
|---|---|
| `rrsi.json` | RRSI's hyperparameters: the coding instance's (k 2, m 2, b 1..4, w 3, m_draft 1, n_prune 4, beta0 0.10, w_s 0, w_c 15, w_n 0.5, five repairs, 30 + 6 traces) with beta1 = 25 (25% tokens per additional passing trial of 100) and T = 10 rounds (a budget choice, as Meta-Harness Adapter runs 6 rounds here); delta measured from H_0's evaluation |
| `domain.py` | OfficeQA as RRSI's domain: the critic's patterns (question ids, bulletin file names, the answers and the scorer), plus the training answers the judge's feedback has revealed (each whole answer of four or more characters, and each number in one with three or more significant digits; bare years and round numbers are left to the LLM critic), the components the agent can carry (no hooks or roles here), the smoke tasks (two questions the incumbent answered, one trial each) |
| `texts/` | the constitution (`SKILL.md`), the pattern library (`PATTERNS.md`) and the four roles' domain paragraphs (`briefs.json`), adapted from upstream's coding and workspace domains |
| `trials.py` | the acting agent and the judge on a trial row: `<id>#<k>` runs and scores as the question `<id>` |
| `task-info/README.md` | the task and how the agent receives the adapter, for the proposer |

Each step's `step_NNNN/output/report.md` says what the step recorded, decided and handed back; `r<t><v>/` there holds
a variant's proposer sessions. `workspace/state/rrsi/` holds upstream's records (the frontier, delta's calibration,
the edit history, the attribution of each edit's predictions, and per round the analysis, the directives, each
variant's proposal, critic verdicts, diff and smoke, and the decisions); `workspace/state/candidates/<id>/adapter/`
each candidate's files.

## Result

One training (`run_luna.yaml`, run `luna-rrsi`, 2026-11-01) and one evaluation of its final adapter H_T on the 172 test
questions, with H_0 evaluated the same way in the same window; Codex 0.164.0 with GPT-5.6-Luna (medium) acting and
learning, 3,600 s per session; exact match. One pass, so each accuracy carries a standard error of about 3.7 points.

On the evolve set (50 questions x 2 trials) H_0 scored 0.41 and upstream's calibration put delta at 0.067 (about 7
of the 100 trials; its within-task bootstrap reads low by about sqrt(2) with two trials, as upstream's does). Ten
rounds of two candidates: 15 were measured, 4 were accepted, and the 5 exploration variants of the stalled rounds (4,
6-9) never reached evaluation. In rounds 6-9, 22 of their 24 verdicts were the reserved-slot check, not the critic's
judgement: the template's guide described components (a procedure in prose, settings, notes) that the tag check does
not read, so the slot was out of reach and those rounds had one real candidate each. This was fixed after the run
(the guide now names what each tag needs in the diff, and a domain declares the components its agent can carry);
see the note at the end.

| Round | Accepted candidate | S (50 x 2) | dS | dC | Why it was admissible |
|---|---|---|---|---|---|
| 0 | `r00B`: an answer-serialization rule and a pre-calculation specification of the measure, period and convention in `ADAPTER.md` | 0.63 | +0.22 | +0.9% | dS > delta, within the cost rule (A, 0.62, also admissible) |
| 1 | `r01B` | 0.62 | -0.01 | -1.0% | inside the band, fewer tokens |
| 5 | `r05A` | 0.63 | +0.01 | -6.7% | inside the band, fewer tokens |
| 7 | `r07A` = H_T | 0.66 | +0.03 | -6.2% | inside the band, fewer tokens |

The three in-band acceptances cut tokens by less than twice the null spread of two evaluations of the same adapter
(sd of dC about 8.7%, from the unbiased within-task variance of H_0's two trials per question): with the coding
instance's weights (w_s = 0) an in-band candidate is admitted for any token saving, so these are noise-level choices,
as the method defines them. Candidates that
scored higher inside the band but cost tokens were refused (rounds 2 and 3: 0.67 at +1.5% and +10%), and so were new
structural components that did not pay for their tokens (a helper script at +41%, a skill at +15%, notes at +3.4%).

| Test (172) | accuracy [95% bootstrap] |
|---|---|
| H_0, SkillOpt's initial skill, this harness, the same window | 0.378 [0.308, 0.454] |
| **RRSI Adapter, H_T** | **0.593** [0.523, 0.663] |

Paired per question with 95% bootstrap intervals, against the GPT-5.6-Luna column: RRSI Adapter - H_0 **+21.5**
[+14.0, +29.7] (46 won, 9 lost); - GEPA -0.7 [-5.1, +3.5]; - SkillOpt +4.7 [-0.1, +9.4]; - Harness² -1.7 [-8.1,
+4.1]; - ReHarness **+7.6** [+1.2, +14.0]; - Continual Harness **+11.6** [+4.7, +18.6]. All of the gain over H_0 is
the answer's form, as in the rest of the column: checked answer by answer, the questions answered with the right
value in another form fall from 52 to 14, and the share with the right value stays at 0.68 (0.680 -> 0.674; paired on
the value alone -0.6 [-5.8, +4.7], 10 won, 11 lost). H_T used 8% fewer tokens per question than H_0 on average (170 K
against 185 K; paired 95% interval -18% to +3.5%; fewer on 83 of 172 questions, the median question 2% more), not
distinguishable from no change.

H_T is `ADAPTER.md` alone (5.2 KB, from 0.9 KB): the starting skill with a pre-calculation specification of the
measure, period and convention, `<answer>` treated as a serialization step, a bounded operand ledger checked against
one contiguous source excerpt, a bounded recovery branch for missing tables, and a written input-operation-output
mapping computed twice. It names no question, file or value.

Spending and time: 16 evaluation steps (1,600 rollouts), 317 M tokens; the proposer and the search roles 218 M
tokens (56 proposer sessions, counting the resumed ones that fixed a done.json or answered the critic, and the
analyst's, digesters' and critic's calls, each one Codex session) and the 30 smoke rollouts 2.5 M (together the
Evolver's 221 M); 5.5 h; the two test evaluations 7 and 9 min. All on the Codex plan, no API key.

This run used the Evolver as it was before an independent review of the integration on 2026-11-01, which found and
fixed: the exploration slot above; proposer sessions that ran from the workspace root (a few relative writes were lost
and cost repair rounds); variants that could read each other's session logs; the critic's diff taken from a git
repository inside the candidate instead of from the stored files; no check against symbolic links in a candidate; the
digesters' read-only shell commands running outside the run's sandbox; and an unbounded re-hand-back. None of them
changed an admission decision in this run (recomputing upstream's selection from the stored measurements gives the
same winner in every round), and the leakage audit found no question, file or revealed answer in any stored
candidate. The code each run used is now recorded in `workspace/state/rrsi/code.jsonl`.
