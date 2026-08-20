from __future__ import annotations

import asyncio
import os
from pathlib import Path

from .base import ExecResult, Sandbox, _to_str
from .registry import register_sandbox


class _LocalShell:
    """Long-lived host shell (a persistent ``bash`` subprocess).

    cwd / env persist across :meth:`run` calls, matching the sandbox ``open_shell``
    contract consumed by :mod:`uni_agent.tools.shell` (no tmux dependency).
    """

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None

    async def run(self, command: str, *, timeout: float | None = None) -> ExecResult:
        if self._proc is None or self._proc.returncode is not None:
            raise RuntimeError("local shell not started")
        # Unique delimiters so streamed output can be split reliably.
        marker = f"__UA_END_{os.getpid()}_{id(self)}__"
        wrapped = f"{{ {command}\n }}; echo {marker} $?"
        self._proc.stdin.write(wrapped.encode() + b"\n")
        await self._proc.stdin.drain()
        chunks: list[str] = []
        exit_code = -1
        while True:
            line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=timeout)
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            if text.startswith(marker):
                code = text[len(marker):].strip()
                try:
                    exit_code = int(code)
                except ValueError:
                    exit_code = -1
                break
            chunks.append(text)
        return ExecResult(exit_code=exit_code, stdout="\n".join(chunks), stderr="")

    async def close(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass
            try:
                await self._proc.wait()
            except Exception:
                pass
        self._proc = None


@register_sandbox("local")
class LocalSandbox(Sandbox):
    """Runs commands on the host via ``asyncio`` subprocesses (no container).

    File operations use the host filesystem directly. Constructed with no args,
    so it uses the base :meth:`Sandbox.from_config` (which ignores the config
    fields).
    """

    supports_shell = True

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def open_shell(
        self,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> _LocalShell:
        """Open a persistent host ``bash`` subprocess (cwd/env persist across runs)."""
        merged_env = {**os.environ, **(env or {})}
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "--norc",
            "--noprofile",
            "-i",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=merged_env,
        )
        shell = _LocalShell()
        shell._proc = proc
        # bash -i on a non-tty pipe prints job-control warnings at startup
        # ("cannot set terminal process group", "no job control"); drain them now
        # so they never pollute the first observation seen by the policy.
        await asyncio.sleep(0.2)
        while True:
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=0.5)
            except asyncio.TimeoutError:
                break
            if not line:
                break
        return shell

    async def read_file(self, path: str) -> bytes:
        """Read directly from the host filesystem without base64 transport."""
        return await asyncio.to_thread(Path(path).read_bytes)

    async def write_file(self, path: str, content: bytes | str) -> None:
        """Write directly to the host filesystem, creating parent directories."""
        data = content.encode("utf-8") if isinstance(content, str) else content
        target = Path(path)

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        await asyncio.to_thread(_write)

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        import os

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            env={**os.environ, **env} if env else None,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return ExecResult(exit_code=-1, stdout="", stderr=f"local exec timed out after {timeout}s")
        return ExecResult(exit_code=proc.returncode or 0, stdout=_to_str(out), stderr=_to_str(err))
