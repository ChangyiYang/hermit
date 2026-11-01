"""OfficeQA as an RRSI Adapter domain: the texts (texts/: the constitution and the four roles' briefs, adapted from
upstream's coding and workspace domains), the critic's patterns, the answers the judge's feedback reveals (added to
the critic's denylist as they are seen), and the smoke tasks."""
from __future__ import annotations

from pathlib import Path
import re

from hermit.algorithms.evolvers.rrsi_adapter.domain import AdapterDomain

HERE = Path(__file__).resolve().parent
EXPECTED = re.compile(r"but expected '([^']*)'")


class OfficeQADomain(AdapterDomain):
    name = "officeqa"
    texts = HERE / "texts"
    # With `deploy: adapter` the agent runs no hooks and registers no roles (task-info/README.md): control_flow,
    # output_plumbing, context_mgmt and subagent have no carrier here, so no reserved slot asks for them.
    carriers = ["prompt", "config", "client_tool", "skill", "memory"]
    critic_patterns = [
        (r"\bUID\d{4}\b", "question id in diff"),
        (r"treasury_bulletin_\d{4}_\d{2}", "a specific bulletin file name in diff"),
        (r"answers\.jsonl|/\.private/|utils/judge\.py|skillopt\.envs\.officeqa|normalize_answer|exact_match",
         "answers, judge or scoring code referenced in diff"),
    ]

    def leak_patterns(self, records: list[dict]) -> list:
        """The training answers the judge's feedback has revealed, as patterns over the diff's added lines: each
        whole answer of four or more characters with a digit, and each number inside an answer with three or more
        significant digits (thousands separators optional), so an answer copied into the adapter, whole or in part,
        is rejected before measurement. Bare years and short numbers are left to the LLM critic: they occur in
        general procedure too. The reason names the value (the proposer wrote it; nothing new is revealed)."""
        answers = set()
        for record in records:
            m = EXPECTED.search(str((record.get("judge") or {}).get("feedback") or ""))
            if m:
                answers.add(m.group(1).strip())
        out, seen = [], set()
        for value in sorted(answers):
            if len(value) >= 4 and re.search(r"\d", value) and not re.fullmatch(r"(19|20)\d\d", value) \
                    and value not in seen:
                seen.add(value)
                out.append((rf"(?m)^\+.*(?<![\w.]){re.escape(value)}(?![\w.])",
                            f"a training answer in diff: {value!r}"))
            for number in re.findall(r"\d[\d,]*\.?\d*", value):
                digits = re.sub(r"\D", "", number).strip("0")       # significant digits, round numbers short
                plain = number.replace(",", "").rstrip(".")
                if len(digits) < 3 or re.fullmatch(r"(19|20)\d\d", plain) or plain in seen:
                    continue
                seen.add(plain)
                whole, _, frac = plain.partition(".")
                grouped = re.sub(r"\B(?=(\d{3})+$)", ",?", whole)
                pattern = grouped + (r"\." + re.escape(frac) if frac else "")
                out.append((rf"(?m)^\+.*(?<![\d.,]){pattern}(?![\d])", f"a number from a training answer in diff: "
                                                                          f"{plain!r}"))
        return out

    def smoke_ids(self, incumbent_per_task: dict | None = None) -> list[str]:
        """Upstream's coding smoke runs two fixed tasks once each; here the first row of the first two questions the
        incumbent answered right in every trial (else the first two questions)."""
        per_task = incumbent_per_task or {}
        solved = [t for t, tr in sorted(per_task.items()) if tr.rewards and min(tr.rewards) >= 1.0]
        chosen = (solved or sorted(per_task))[:2]
        return [((per_task[t].extra or {}).get("rows") or [t])[0] for t in chosen]
