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
hermit eval  --config run_luna.yaml --adapter seed_adapter   # H_0 the same way, in the same window
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

Four trainings with this Evolver (`run_luna.yaml`, runs `luna-rrsi-r2` to `luna-rrsi-r5`, 2026-11-01), one with the
engineering instance's selection weights (`run_luna_eng.yaml`, run `luna-rrsi-eng`) and the first training
(`luna-rrsi`, made with the Evolver as it was before an independent review; see the note at the end). Each final
adapter H_T was evaluated once on the 172 test questions, and H_0 once in the first training's window; Codex 0.164.0
with GPT-5.6-Luna (medium) acting and learning, 3,600 s per session; exact match. The four trainings, the engineering
arm and their test evaluations ran with `agent.concurrency=6` on the command line (five trainings side by side on one
login); the first training and H_0's evaluation ran with the configs' 10. A one-pass accuracy carries a standard error
of about 3.7 points.

| Test (172), one pass each | accuracy [95% bootstrap] |
|---|---|
| H_0, SkillOpt's initial skill, this harness | 0.378 [0.308, 0.448] |
| **RRSI Adapter, H_T, mean of four trainings** | **0.602** [0.538, 0.663]: 0.605, 0.581, 0.599, 0.622 (sd 0.017) |
| RRSI Adapter, the engineering instance's selection weights, one training | 0.616 [0.547, 0.686] |
| RRSI Adapter, the first training (before the review) | 0.593 [0.517, 0.663] |

Paired per question with 95% bootstrap intervals, the four trainings averaged per question first as the column
averages GEPA's and SkillOpt's four, against the GPT-5.6-Luna column: RRSI Adapter − H_0 **+22.4** [+15.7, +29.1] (64
won, 18 lost); − GEPA +0.1 [−3.2, +3.6]; − SkillOpt **+5.5** [+1.0, +10.0]; − Harness² −0.9 [−6.1, +4.5]; −
Meta-Harness Adapter −0.3 [−4.7, +4.1]; − ReHarness **+8.4** [+2.9, +14.4]; − Continual Harness **+12.5** [+6.2,
+18.9]. The first training lies among the four (−0.9 [−5.7, +3.9] against their mean): the review's fixes did not move
the result. All of the gain over H_0 is the answer's form, as in the rest of the column. Every non-exact answer was
read for whether its value is right (each read as right, or where the reading and an automatic match disagreed, read
again independently: the same verdict all 83 times): the share of answers with the right value is 0.686 for H_0 and
0.693 over the four trainings (0.663–0.709 each); paired on the value alone, the four trainings − H_0 is +0.7 [−3.5,
+4.9]. What changes is the form: the answers with the right value in a form the judge rejects fall from 53 to 16 per
training on average.

On the evolve set (the 50 training questions, two trials each):

| Run | delta | H_0 | candidates measured / refused | accepted (inside the noise band) | H_T: S, files | rollouts; tokens of the rollouts + the Evolver; time |
|---|---|---|---|---|---|---|
| `luna-rrsi-r2` | 0.063 | 0.42 | 20 / 0 | 4 (3) | `r06B` 0.61: `ADAPTER.md` and two helper scripts, 9.6 KB | 2,100; 465 M + 165 M; 7.3 h |
| `luna-rrsi-r3` | 0.052 | 0.41 | 20 / 0 | 3 (2) | `r03A` 0.67: `ADAPTER.md` and a statistics script, 8.0 KB | 2,100; 405 M + 151 M; 6.5 h |
| `luna-rrsi-r4` | 0.045 | 0.45 | 20 / 0 | 3 (2) | `r08B` 0.62: `ADAPTER.md`, 3.8 KB | 2,100; 401 M + 134 M; 6.3 h |
| `luna-rrsi-r5` | 0.066 | 0.43 | 20 / 0 | 2 (1) | `r07B` 0.66: `ADAPTER.md`, 4.0 KB | 2,100; 397 M + 149 M; 6.6 h |
| `luna-rrsi-eng` | 0.053 | 0.45 | 20 / 0 | 2 (1) | `r03A` 0.71: `ADAPTER.md`, 2.2 KB | 2,100; 395 M + 162 M; 7.0 h |
| `luna-rrsi` (before the review) | 0.067 | 0.41 | 15 / 5 | 4 (3) | `r07A` 0.66: `ADAPTER.md`, 5.2 KB | 1,600; 317 M + 221 M; 5.5 h |

- **Round 0 makes the gain.** In every training the first round's accepted candidate rose 0.20–0.27 above H_0 on the
  evolve set with prompt edits (above all, `<answer>` as a serialization step: only the requested value or values, in
  the requested shape, rounding, sign and scale); every later acceptance is inside the noise band.
- **The exploration slot works.** Every variant of every round reached evaluation (20 of 20, against 15 of 20 in the
  first training, where the exploration variants of rounds 6–9 failed the tag check and round 4's was refused by the
  critic), and each training measured every component the agent can carry (`prompt`, `config`, `client_tool`, `skill`,
  `memory`). Helper scripts entered two final adapters, admitted inside the band for their novelty: `luna-rrsi-r2`'s
  `scripts/show_context.sh` (a numbered window of a corpus file) and `scripts/calc_fallback.sh` (a calculator), which
  the acting agent ran in about 163 and 90 of the 172 test sessions, and `luna-rrsi-r3`'s `scripts/stats.py`, run in
  13.
- **Selection inside the band.** With the coding instance's weights (w_s = 0) a candidate inside the band is admitted
  for any token saving, or for a new structural component (`client_tool`, `skill`, `memory`) that adds less than about
  3% tokens (w_n / w_c = 0.5 / 15), so H_T can score below the best candidate measured: 0.61 where 0.69 was measured
  (`luna-rrsi-r2`), 0.62 where 0.69 was (`luna-rrsi-r4`), 0.66 where its own round-0 candidate scored 0.70
  (`luna-rrsi-r5`). The engineering weights (w_s 100, w_c 2: inside the band the score counts) kept the best candidate
  measured, 0.71, and its H_T scores 0.616 on the test questions, +1.5 [−3.5, +6.2] against the four trainings' mean:
  one training cannot tell the two weightings apart.
- **Tokens per test question.** H_0 185 K; the four H_T 241, 167, 182 and 170 K. `luna-rrsi-r2`'s round-0 candidate
  cost 26% more tokens on the evolve set for +0.23, inside the cost rule (dC <= beta0 + beta1 dS), and its H_T uses
  30% more on the test questions [+16%, +46%]; the other three are not distinguishable from H_0, nor is the mean of the
  four (+2.7% [−6.3%, +12.2%]).
- **No leakage.** No final adapter names a question, a bulletin file, or a training or test answer (checked for
  question ids, file names, whole answers of four or more characters and numbers of three or more significant digits).

Spending: per training 2,100 evaluation rollouts (395–465 M tokens) and 134–165 M tokens for the proposer, the search
roles and the smoke rollouts, 6.3–7.3 h with the five trainings side by side; their test evaluations 11–25 min. The
five trainings and their evaluations took the plan's weekly window from 62% to 74% (whole percents, account-wide: an
upper bound on their share). All on the Codex plan, no API key.

The first training (`luna-rrsi`) used the Evolver as it was before an independent review of the integration on
2026-11-01, which found and fixed: the exploration slot (in rounds 6–9, 22 of variant B's 24 verdicts were the
reserved-slot tag check, not the critic's judgement: the template's guide described components the tag check does not
read); proposer sessions that ran from the workspace root (a few relative writes were lost); variants that could read
each other's session logs; the critic's diff taken from a git repository inside the candidate instead of from the
stored files; no check against symbolic links in a candidate; the digesters' read-only shell commands running outside
the run's sandbox; and an unbounded re-hand-back. None of them changed an admission decision in that run (recomputing
upstream's selection from its stored measurements gives the same winner in every round), and its H_T (`ADAPTER.md`
alone) names no question, file or revealed answer. The code each later run used is recorded in
`workspace/state/rrsi/code.jsonl`: the four trainings and the engineering arm ran this Evolver.
