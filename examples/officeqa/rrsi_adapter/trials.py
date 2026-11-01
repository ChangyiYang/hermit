"""k trials of a training question as k rows: `make_rows.py` writes `<id>#<k>` rows; the acting agent and the judge
here see the question's own id, so a trial row runs and scores exactly as the question does."""
from __future__ import annotations

from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "utils" / "agent"))
sys.path.insert(0, str(HERE.parent / "utils"))

from codex_agent import OfficeQACodexAgent  # noqa: E402
import judge as _judge  # noqa: E402


def question(row_id: str) -> str:
    return row_id.rsplit("#", 1)[0] if re.fullmatch(r".+#\d+", row_id) else row_id


class TrialCodexAgent(OfficeQACodexAgent):
    """`agent.name: trials.py:TrialCodexAgent`: the OfficeQA Codex agent on a row's question."""

    def run(self, task: dict, adapter):
        return super().run({**task, "id": question(task["id"])}, adapter)


def score(task: dict, output) -> dict:
    """`judge.path: trials.py`: SkillOpt's exact match (../utils/judge.py) on the row's question."""
    return _judge.score({**task, "id": question(task["id"])}, output)
