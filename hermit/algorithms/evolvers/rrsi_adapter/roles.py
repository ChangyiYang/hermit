"""RRSI's one-call search roles (the analyst, its digesters, the critic) on a Codex CLI session each.

Upstream's analyst, digester and critic (rrsi.analyst, rrsi.digester, rrsi.critic) call `generate(prompt, system,
json_only, ...)`, Claude Opus 4.8 on Vertex AI (rrsi.llm). Here `generate` is rebound in those three modules, for the
duration of a call, to `CodexRoles.generate`: one `codex exec` per call, the way the Evolver's proposer is a Codex
session, with the role's system text as the session's developer instructions and the prompt on stdin. The session
runs under the run's sandbox (the workspace's records hidden), Codex's own sandbox read-only, in an empty directory,
so a role answers from the text it is given, as upstream's do; a digester's session runs in the round's rendered
traces instead, the directory upstream's digester reads with its read_file/grep/bash actions. The reply is the session's last message; with
`json_only`, upstream's own `extract_json` cuts the JSON out of it, as upstream's `generate` does.

Everything upstream does with the reply (parsing actions, the digesters' tool loop, the critic's three tries) is
upstream's code, unchanged. Each session's events and reply stay under `log_dir`, and its token usage is summed into
`usage` for the step's report. A refusal for the plan's usage limit stops the call (`Stop`) instead of being retried.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import itertools
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

from hermit.evolver.base import Usage

ROLE_MODULES = ("rrsi.analyst", "rrsi.digester", "rrsi.critic")
# Codex's own words for a refusal on the plan's limit, and the error's codes (codex_login.LIMIT)
LIMIT = re.compile(r"hit your usage limit|usage_limit_reached|usageLimitExceeded", re.I)
UNAVAILABLE = re.compile(r"unexpected status (429|5\d\d)|exceeded retry limit|stream disconnected|too many requests|"
                         r"service unavailable|bad gateway|model is at capacity|server is overloaded", re.I)


class Stop(BaseException):
    """The call must end now: the Trainer cancelled it, or the model's plan refused for its usage limit. A
    BaseException, so that upstream's `except Exception` handlers around a role call never swallow it."""


@contextmanager
def bound(generate, digester_subprocess=None):
    """`generate` in the role modules replaced by `generate` while the block runs, and, when given, the
    `subprocess` module the digester runs its read-only `bash` action with (upstream runs it on the host)."""
    modules = [importlib.import_module(m) for m in ROLE_MODULES]
    saved = [(m, m.generate) for m in modules]
    digester = importlib.import_module("rrsi.digester")
    saved_subprocess = digester.subprocess
    try:
        for m in modules:
            m.generate = generate
        if digester_subprocess is not None:
            digester.subprocess = digester_subprocess
        yield
    finally:
        for m, g in saved:
            m.generate = g
        digester.subprocess = saved_subprocess


class SandboxedSubprocess:
    """What the digester sees as `subprocess` while bound: its `run(cmd, shell=True, cwd=...)` runs the command
    under the run's sandbox with the workspace's records covered, as every other command a learner starts does.
    Upstream's allowlist of read-only commands (rrsi.digester._bash_ok) still applies before it is called.

    The digester runs its shell in the round's rendered traces, which upstream writes under the run's state, and the
    sandbox covers the state: a working directory under a covered path is replaced by `visible`, the copy of those
    traces the digesters' sessions read, so relative paths name the same files."""

    TimeoutExpired = subprocess.TimeoutExpired
    CalledProcessError = subprocess.CalledProcessError

    def __init__(self, sandbox, hide: list[Path], visible: Path | None = None):
        self.sandbox, self.hide, self.visible = sandbox, list(hide), visible

    def _cwd(self, cwd):
        if cwd is None or self.visible is None:
            return cwd
        c = Path(cwd).resolve()
        covered = any(c == h.resolve() or h.resolve() in c.parents for h in self.hide)
        return str(self.visible) if covered else cwd

    def run(self, cmd, shell: bool = False, cwd=None, **kw):
        args = ["bash", "-c", cmd] if shell else list(cmd)
        env = kw.pop("env", None)
        cwd = self._cwd(cwd)
        if self.sandbox is not None:
            from hermit.sandbox import HidingSandbox
            args, env = HidingSandbox(self.sandbox, self.hide).wrap(args, env=dict(env or os.environ))
        return subprocess.run(args, cwd=cwd, env=env, **kw)


def events_of(path: Path) -> list[dict]:
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def usage_of(events: list[dict], model: str) -> Usage | None:
    u = Usage(model=model)
    for e in events:
        if e.get("type") == "turn.completed" and e.get("usage"):
            x = e["usage"]
            u.input_tokens += int(x.get("input_tokens") or 0)
            u.cached_input_tokens += int(x.get("cached_input_tokens") or 0)
            u.output_tokens += int(x.get("output_tokens") or 0)
            u.calls += 1
    return u if u.calls else None


def refusal_of(events: list[dict]) -> str:
    return " ".join(str(e.get("message") or (e.get("error") or {}).get("message") or "")
                    for e in events if e.get("type") in ("error", "turn.failed"))


class CodexRoles:
    """`generate` for upstream's one-call roles, each call one Codex session."""

    def __init__(self, ctx, home: Path, log_dir: Path, model: str | None, args: list[str], timeout_s: float,
                 hide: list[Path] = ()):
        self.ctx, self.home, self.log_dir = ctx, Path(home), Path(log_dir)
        self.model, self.args, self.timeout_s = model, list(args or []), float(timeout_s)
        self.hide = [Path(p) for p in hide]
        self.usage: list[Usage] = []
        self.calls = 0
        self.traces: Path | None = None          # the round's rendered traces, the digesters' working directory
        self._count = itertools.count(1)
        self._lock = threading.Lock()
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def sandboxed_subprocess(self) -> SandboxedSubprocess:
        return SandboxedSubprocess(self.ctx.sandbox, self.hide, visible=self.traces)

    def _cancelled(self) -> bool:
        return self.ctx.cancel is not None and self.ctx.cancel.is_set()

    def generate(self, prompt: str, system: str | None = None, max_retries: int = 6, json_only: bool = False,
                 model: str | None = None, max_tokens: int = 20_000, cache_prefix: str | None = None) -> str:
        """Upstream's `generate` contract: the reply text (JSON cut out with `json_only`); after `max_retries`
        empty or failed sessions, RuntimeError, as upstream raises."""
        from rrsi.llm import _JSON_SUFFIX, extract_json
        system_text = (system or "") + (_JSON_SUFFIX if json_only else "")
        text = f"{cache_prefix}\n\n{prompt}" if cache_prefix else prompt
        last = ""
        for attempt in range(max(1, int(max_retries))):
            if self._cancelled():
                raise Stop("stopped by the Trainer")
            reply, why = self._session(system_text, text)
            if reply.strip():
                return extract_json(reply) if json_only else reply
            last = why
            if LIMIT.search(why):
                raise Stop(f"the model's plan refused: {why[:300]}")
            time.sleep(min(2 ** attempt, 30) * (4 if UNAVAILABLE.search(why) else 1))
        raise RuntimeError(f"generate failed after {max_retries} Codex sessions: {last[:300]}")

    def _session(self, system: str, prompt: str) -> tuple[str, str]:
        with self._lock:
            n = next(self._count)
            self.calls += 1
        call = self.log_dir / f"{n:05d}"
        work = call / "work"
        work.mkdir(parents=True, exist_ok=True)
        (call / "prompt.txt").write_text(prompt)
        (call / "system.txt").write_text(system)
        last = call / "last_message.txt"
        # Upstream's digester reads "the rendered trajectory files in your working directory" through its JSON
        # actions; a Codex session reads its working directory with its own tools as well, so a digester's session
        # runs in the round's rendered traces (Codex's read-only sandbox), and every other role's in an empty one.
        cwd = self.traces if self.traces is not None and "trajectory digester" in system else work
        command = ["codex", "exec", "--json", "--skip-git-repo-check", "--ignore-user-config", "-s", "read-only",
                   "-C", str(cwd), "-c", f"developer_instructions={json.dumps(system, ensure_ascii=False)}",
                   "-c", 'approval_policy="never"', "-c", "features.shell_snapshot=false",
                   "-o", str(last), *(["-m", self.model] if self.model else []), *self.args, "-"]
        env = dict(os.environ)
        env["CODEX_HOME"] = str(self.home)
        sandbox = self.ctx.sandbox
        if sandbox is not None:
            if self.hide:
                from hermit.sandbox import HidingSandbox
                sandbox = HidingSandbox(sandbox, self.hide)
            command, env = sandbox.wrap(command, writable=[self.home, call], env=env, home=self.home)
        events_path = call / "events.jsonl"
        started = time.time()
        with open(call / "prompt.txt", "rb") as stdin, open(events_path, "wb") as out:
            proc = subprocess.Popen(command, stdin=stdin, stdout=out, stderr=subprocess.STDOUT, env=env,
                                    cwd=str(cwd), start_new_session=True)
            timed_out = False
            while proc.poll() is None:
                if self._cancelled() or time.time() - started > self.timeout_s:
                    timed_out = not self._cancelled()
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
                time.sleep(0.5)
            code = proc.wait()
        events = events_of(events_path)
        usage = usage_of(events, self.model or "codex")
        if usage is not None:
            with self._lock:
                self.usage.append(usage)
        reply = last.read_text(errors="replace") if last.is_file() else ""
        if not reply.strip():
            texts = [(e.get("item") or {}).get("text") or "" for e in events if e.get("type") == "item.completed"
                     and (e.get("item") or {}).get("type") == "agent_message"]
            reply = texts[-1] if texts else ""
        (call / "reply.txt").write_text(reply)
        why = refusal_of(events) or ("timed out" if timed_out else
                                     (events_path.read_text(errors="replace")[-600:] if code else ""))
        meta = {"exit": code, "timed_out": timed_out, "wall_s": round(time.time() - started, 1),
                "chars": len(reply), "why": why[:600] if not reply.strip() else ""}
        (call / "meta.json").write_text(json.dumps(meta, indent=1))
        return reply, why

    def total(self) -> list[Usage]:
        """The usage of every session so far, folded into one record."""
        if not self.usage:
            return []
        u = Usage(model=self.model or "codex")
        for x in self.usage:
            u.input_tokens += x.input_tokens
            u.cached_input_tokens += x.cached_input_tokens
            u.output_tokens += x.output_tokens
            u.calls += x.calls
        return [u]
