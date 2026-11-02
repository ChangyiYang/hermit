# OfficeQA

[OfficeQA](https://github.com/databricks/officeqa) ([dataset](https://huggingface.co/datasets/databricks/officeqa),
[paper](https://arxiv.org/abs/2603.08655)): questions about U.S. Treasury Bulletins (1939–2025) whose answers are
figures found in the bulletins, often after a computation. This example runs OfficeQA Full (246 questions) as the
[SkillOpt paper](https://arxiv.org/abs/2605.23904) does: its 50 / 24 / 172 train / validation / test split, each
question's parsed source pages in the prompt, and exact match. It trains the agent's skill, the adapter's
`ADAPTER.md`, with the GEPA and SkillOpt methods, with ReHarness a bank of reflections retrieved below that skill, and
with Meta-Harness Adapter whole adapters written by a coding-agent proposer (Codex harness only), and compares them
with no skill and with SkillOpt's initial skill, in both of the paper's harnesses: direct chat, with read and grep
tools limited to the question's source files, and Codex, a `codex exec` session with the whole corpus. The same
methods are run a second time with Codex and GPT-5.6-Luna acting and learning on a ChatGPT login, the second column
of the [Results](#results) table, where RRSI Adapter joins them.

## Setup

Install the framework first ([Installation](../../docs/get-started/installation.md)) with the `skillopt` and `gepa`
dependency groups (pip 25.1+). Also needed: the gated Hugging Face dataset `databricks/officeqa` (`officeqa_full.csv`,
and `treasury_bulletins_parsed/` with `transformed/` and `jsons/`), and the key of the models' backend: for the default
(GLM 5.3 Flash acting, GLM 5.3 learning, on Fireworks) `FIREWORKS_API_KEY`. `model=<backend>/<model id>` on the
command line moves the agent, `llm=...` the learner ([Model backends](../../docs/guides/model-backends.md)). The Codex
harness also needs Docker, set up once as in the [recipe](../../docs/recipes/officeqa.md#run-in-the-codex-harness).

```bash
python -m pip install -e . --group skillopt --group gepa
python examples/officeqa/utils/prepare.py --csv /path/to/officeqa_full.csv --docs /path/to/treasury_bulletins_parsed
```

`prepare.py` writes the questions, the paper's split, SkillOpt's initial skill (`utils/seed_adapter/`) and a link to the
corpus (`utils/documents`, what the agent's sandbox is granted) to `utils/`.

## Run

The two columns of [Results](#results) are two sets of configs: each method's `run_codex_flash.yaml` or
`stream_codex.yaml` (GLM 5.3 Flash acting and learning) and its `run_codex_luna.yaml` or `stream_codex_luna.yaml`
(GPT-5.6-Luna acting and learning), the same protocol with the models changed. Both run in the Codex harness, which
needs Docker, set up once as in the [recipe](../../docs/recipes/officeqa.md#run-in-the-codex-harness).

#### GLM 5.3 Flash with Codex CLI 0.162.0

```bash
export OFFICEQA_CODEX_IMAGE=officeqa-codex:0.162.0
hermit train --config examples/officeqa/gepa/run_codex_flash.yaml                                        # GEPA
hermit train --config examples/officeqa/skillopt/run_codex_flash.yaml                                    # SkillOpt
hermit eval  --config examples/officeqa/gepa/run_codex_flash.yaml     --adapter final                    # GEPA's skill
hermit eval  --config examples/officeqa/skillopt/run_codex_flash.yaml --adapter final                    # SkillOpt's skill
hermit eval  --config examples/officeqa/skillopt/run_codex_flash.yaml --adapter none                     # no skill
hermit eval  --config examples/officeqa/skillopt/run_codex_flash.yaml --adapter ../utils/seed_adapter    # initial skill
python examples/officeqa/meta_harness_adapter/make_seed.py                                               # Meta-Harness Adapter
hermit train --config examples/officeqa/meta_harness_adapter/run.yaml                                    # six rounds of two candidates
hermit eval  --config examples/officeqa/meta_harness_adapter/run.yaml --adapter final
python examples/officeqa/continual_harness/make_seed.py                                                  # Continual Harness
hermit train --config examples/officeqa/continual_harness/stream_codex.yaml                              # the 172 as one stream
FIREWORKS_API_KEY=... WM_PORT=3601 examples/officeqa/reharness/launch.sh stream_codex                    # ReHarness: the same stream
FIREWORKS_API_KEY=... SEED=0 WM_PORT=3591 examples/officeqa/harness2_stream/launch.sh                    # Harness²: the same stream
```

An adapter's accuracy is `mean_score` in `<method>/runs/<run name>/eval/<selector>/summary.json` (172 test
questions, one pass), where `<selector>` is `final`, `none`, `adapter` or a directory's name such as `seed_adapter`;
`--adapter adapter` evaluates the trained adapter included in the folder. A stream has no test split: its number is
the mean of `rollout.mean_score` over the steps of its run's `steps.jsonl` (step 0, the bootstrap, has none). A
smoke on a few questions: the folder's `smoke_codex.yaml`, after `hermit check --config ...`
([walkthrough](../../docs/recipes/officeqa.md#check-train-evaluate)).

#### GPT-5.6-Luna with Codex CLI 0.164.0

Each `*_luna.yaml` is the Flash-row config above with the models changed and nothing else. The acting model runs on
Codex's built-in provider with the machine's ChatGPT login (`codex login` once, no API key:
[`utils/agent/codex_login.py`](utils/agent/codex_login.py) gives each session a copy of the login), the learner is
the same model behind a relay that answers each `llm` call with one Codex session
([`utils/agent/codex_chat.py`](utils/agent/codex_chat.py)), and Harness²'s editors are Codex sessions
([`harness2_stream/codex_proposer.py`](harness2_stream/codex_proposer.py)). The image with the machine's own Codex
and the egress proxy to chatgpt.com: the recipe's [A model on a ChatGPT login](../../docs/recipes/officeqa.md#a-model-on-a-chatgpt-login).

```bash
export OFFICEQA_CODEX_IMAGE=officeqa-codex:0.164.0 OFFICEQA_CODEX_PROXY=http://oq-egress-chatgpt:3128
python examples/officeqa/utils/agent/codex_chat.py --port 3671 &                                         # the learners' relay
hermit train --config examples/officeqa/gepa/run_codex_luna.yaml                                         # GEPA
hermit train --config examples/officeqa/skillopt/run_codex_luna.yaml                                     # SkillOpt
hermit eval  --config examples/officeqa/gepa/run_codex_luna.yaml     --adapter final                     # GEPA's skill
hermit eval  --config examples/officeqa/skillopt/run_codex_luna.yaml --adapter final                     # SkillOpt's skill
hermit eval  --config examples/officeqa/skillopt/run_codex_luna.yaml --adapter none                      # no skill
hermit eval  --config examples/officeqa/skillopt/run_codex_luna.yaml --adapter ../utils/seed_adapter     # initial skill
python examples/officeqa/continual_harness/make_seed.py                                                  # Continual Harness
hermit train --config examples/officeqa/continual_harness/stream_codex_luna.yaml                         # the 172 as one stream
FIREWORKS_API_KEY=... WM_PORT=3601 examples/officeqa/reharness/launch.sh stream_codex_luna               # ReHarness (embeddings on Fireworks)
CONFIG=run_codex_luna.yaml examples/officeqa/harness2_stream/launch.sh                                   # Harness²: the editors as Codex sessions
```

## Results

<table>
<thead>
<tr><th rowspan="2">Method</th><th colspan="2">Accuracy</th></tr>
<tr><th>GLM 5.3 Flash</th><th>GPT-5.6-Luna</th></tr>
</thead>
<tbody>
<tr><td>no skill</td><td>0.541</td><td>0.308</td></tr>
<tr><td>initial skill</td><td>0.576</td><td>0.360</td></tr>
<tr><td><b>GEPA</b></td><td>0.580</td><td>0.600</td></tr>
<tr><td><b>SkillOpt</b></td><td>0.600</td><td>0.546</td></tr>
<tr><td><b>Meta-Harness Adapter</b></td><td>0.640</td><td>0.605</td></tr>
<tr><td><b>RRSI Adapter</b></td><td>not run</td><td>0.602</td></tr>
<tr><td><b>Continual Harness</b></td><td>0.634</td><td>0.477</td></tr>
<tr><td><b>ReHarness</b></td><td>0.634</td><td>0.517</td></tr>
<tr><td><b>Harness²</b></td><td>0.651</td><td>0.611</td></tr>
</tbody>
</table>

Results use two settings:

- **Train/test split:** [GEPA](../../docs/methods/gepa.md) and [SkillOpt](../../docs/methods/skillopt.md) train one
  `ADAPTER.md` from SkillOpt's initial skill on 50 training questions, select on 24 validation questions, and evaluate
  the fixed final adapter on 172 test questions. Accuracies are means of four independent trainings, with one
  evaluation pass per final adapter. Configs: [`gepa/run_codex_flash.yaml`](gepa/run_codex_flash.yaml) and
  [`skillopt/run_codex_flash.yaml`](skillopt/run_codex_flash.yaml). [Meta-Harness
  Adapter](../../docs/methods/meta_harness_adapter.md) searches on the 50 training questions alone (no validation
  questions) and evaluates its best candidate once on the 172; one training, in the Codex harness the streams use
  ([`meta_harness_adapter/run.yaml`](meta_harness_adapter/run.yaml), [result](meta_harness_adapter/#result)).
  [RRSI Adapter](../../docs/methods/rrsi_adapter.md) evolves the adapter on the 50 training questions with two trials
  each (no validation questions), ten rounds of two candidates under RRSI's regularizers, and evaluates the final
  incumbent once on the 172; the mean of four trainings, GPT-5.6-Luna only, in the same Codex harness
  ([`rrsi_adapter/run_luna.yaml`](rrsi_adapter/run_luna.yaml), [result](rrsi_adapter/#result)).
- **Test streaming:** [Continual Harness](../../docs/methods/continual_harness.md) and
  [Harness²](../../docs/methods/harness2.md) each process the 172 test questions once in the same order,
  updating the adapter as they go, without a separate training split or access to judge verdicts. These
  scores measure performance during adaptation. See [`continual_harness/`](continual_harness/) and
  [`harness2_stream/`](harness2_stream/) for their protocols.

**ReHarness** ([`reharness/`](reharness/), [method](../../docs/methods/reharness.md)) keeps the initial skill as
the harness's text and learns a bank of reflections below it, five retrieved per question by Qwen3-Embedding-8B and
revised after it, with the acting model as its reflection and revision agents. Each session is Codex in a container
of its own through [`utils/agent/codex_agent.py`](utils/agent/codex_agent.py) (3,600 s wall clock, as the two
streams; GEPA's and SkillOpt's trainings used SkillOpt's own Codex harness with 180 s sessions), which deploys the
skill and the question's retrieved reflections as the session's `AGENTS.md`. Five retrieved reflections run past
Codex's default limit on `AGENTS.md`, 32 KiB, beyond which Codex silently drops the rest, so the ReHarness configs
raise it to 200 KiB (`project_doc_max_bytes: 204800`); no session's `AGENTS.md` reached it.

- **Test streaming** ([`reharness/stream_codex.yaml`](reharness/stream_codex.yaml)): the 172 in the same seed-0
  order as Continual Harness and Harness², each answered by the bank as it stands, no verdict to the learner
  (`feedback: false`: the reflection and revision agents read the trajectory and the answer, never the judge's
  verdict or the reference answer). One stream on 2026-10-28: **0.634**, 0.640 over the first 86 questions and
  0.628 over the last 86; paired against Harness²'s reference execution of the initial skill (0.576), +5.8 points
  [−1.2, +13.4] (26 questions won, 16 lost). 10.0 h (USD 2.43 of sessions, USD 4.81 of reflection and revision
  calls), no errors, no session timed out; 172 reflections of 8.8 K characters on average.

Paired differences with 95% intervals, each run's selection and cost, the Codex sessions' isolation and the
differences from the paper: the [recipe](../../docs/recipes/officeqa.md#the-flash-row).

The earlier direct-chat runs (`run.yaml`, `stream.yaml`: SkillOpt's rollout, three passes) are in the recipe's
[direct-chat table](../../docs/recipes/officeqa.md#results); they are not on this row.

The GPT-5.6-Luna column is the same methods with the model changed and nothing else of the protocol: Codex 0.164.0
with GPT-5.6-Luna acting and learning on the machine's ChatGPT login instead of GLM 5.3 Flash on Fireworks, the same
data, split, seed skill, judge, parameters, stream order and one pass (Meta-Harness Adapter's run on this column came
later, `meta_harness_adapter/run_codex_luna.yaml`; RRSI Adapter ran on this column only). It starts 22–23 points lower
and gains more from every method, most from GEPA, Harness² and Meta-Harness Adapter, and almost all of the gain is the
answer's form: a check of every answer finds the share with the right value at 0.66–0.71 in every cell of the column.
Meta-Harness Adapter and RRSI Adapter, which search whole adapters in the Codex harness the streams use, reach GEPA
and Harness² too, from 0.378 for their starting adapter in that harness (Meta-Harness Adapter − GEPA +0.4 [−3.9,
+4.7]; RRSI Adapter, its four trainings averaged per question, − GEPA +0.1 [−3.2, +3.6], − Meta-Harness Adapter −0.3
[−4.7, +4.1], − its start +22.4 [+15.7, +29.1], all of it the answer's form: checked answer by answer, the share with
the right value is 0.686 for the start and 0.693 for the four). A Meta-Harness Adapter training used 650 rollouts and
3 M learner tokens, an RRSI Adapter training 2,100 rollouts and 134–165 M. What differs from the GLM column, the
intervals, the paired differences and the check: the
[recipe](../../docs/recipes/officeqa.md#the-flash-row-with-gpt-56-luna); the `*_luna.yaml` configs run it
([Run](#run)).

## Code

```text
utils/agent/        OfficeQAAgent (agent.py): SkillOpt's rollout, direct or Codex, with ADAPTER.md as the skill;
                    codex_sandbox.py and docker/ are the Codex harness's container, image and egress proxy
                    (Dockerfile.standalone: the image with a machine's standalone Codex); codex_login.py puts a
                    model on a ChatGPT login (per-session copies of the login, the plan's usage, session slots);
                    codex_chat.py is the learners' relay, one Codex session per chat-completions call
utils/prepare.py    tasks.jsonl, the splits, data.json and seed_adapter/      utils/judge.py   SkillOpt's exact match
utils/task-info/    the task for the learner
gepa/ skillopt/     smoke.yaml and run.yaml per method, smoke_codex.yaml and run_codex.yaml for the Codex harness,
                    run_codex_flash.yaml and run_codex_luna.yaml for the two columns of Results; adapter/ is the
                    skill run.yaml trained (direct chat), with its .hermit/CARD.md
harness2_stream/    Harness² under the paper's protocol, by the upstream code, with Codex as the acting agent
                    (run_codex.yaml, slots_codex.yaml; utils/agent/codex_agent.py and codex_session.py; the
                    harness2-codex template); launch.sh starts a stream; run_codex_luna.yaml and codex_proposer.py:
                    the editors as Codex sessions on a ChatGPT login (CONFIG=run_codex_luna.yaml launch.sh)
reharness/          ReHarness: stream_codex.yaml (the 172 test questions as one stream, Codex as the acting agent;
                    stream_codex_luna.yaml the same on the ChatGPT login),
                    run_codex_flash.yaml (the 50 as one stream, then the frozen bank on the 172), the direct-chat
                    smoke.yaml, run.yaml and stream.yaml; launch.sh starts the embedding relay and the command;
                    make_seed.py writes seed_adapter/, the initial skill over the reharness template
rrsi_adapter/       RRSI Adapter on the GPT-5.6-Luna column: run_luna.yaml and smoke_luna.yaml, rrsi.json (RRSI's
                    hyperparameters), run_luna_eng.yaml with rrsi_eng.json (the engineering instance's selection
                    weights), domain.py and texts/ (the benchmark's rules and the roles' texts, adapted from
                    upstream's), trials.py and make_rows.py (two trials per training question as rows <id>#k),
                    make_seed.py
```
