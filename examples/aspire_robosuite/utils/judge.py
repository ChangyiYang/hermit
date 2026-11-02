"""The held-out success rate of the selected program: the fraction of evaluation seeds on which ASPIRE's harness
reports task_success (Robosuite's own success check, run by the host after the program finished)."""


def score(task: dict, output) -> dict:
    extra = output.extra or {}
    if extra.get("role") == "verifier":
        return {"score": float(output.output),
                "feedback": f"verifier: {extra.get('debug_passed')}/{extra.get('debug_total')} seeds with "
                            f"{len(extra.get('versions') or [])} attempt(s)"}
    return {"score": float(output.output),
            "feedback": f"{extra.get('eval_passed')}/{extra.get('eval_total')} held-out seeds "
                        f"({extra.get('debug_passed')}/{extra.get('debug_total')} development, {extra.get('selected')})",
            "debug_rate": (extra.get("debug_passed") or 0) / max(1, extra.get("debug_total") or 1)}
