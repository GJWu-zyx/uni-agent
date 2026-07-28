from typing import Any

import pytest

from uni_agent.sandbox import ExecResult
from uni_agent.tools.shell import ShellChannel


class RecordingBackend:
    def __init__(self) -> None:
        self.exec_calls: list[list[str]] = []
        self.writes: list[tuple[str, bytes | str]] = []

    async def exec(self, argv: list[str], **kwargs: Any) -> ExecResult:
        self.exec_calls.append(list(argv))
        return ExecResult(exit_code=0, stdout="", stderr="")

    async def write_file(self, path: str, content: bytes | str) -> None:
        self.writes.append((path, content))


@pytest.mark.asyncio
async def test_start_command_keeps_large_payload_out_of_tmux_argv():
    backend = RecordingBackend()
    channel = ShellChannel(backend, session_id="test-session")  # type: ignore[arg-type]
    command = "printf 'large payload\\n'\n" * 10_000

    command_id = await channel.start_command(command)

    command_path = "/tmp/uni-agent-shell/test-session/cmd_1.input"
    assert command_id == 1
    assert backend.writes == [(command_path, command)]
    assert len(backend.exec_calls) == 1

    send_keys_argv = backend.exec_calls[0]
    injected_line = send_keys_argv[-2]
    assert send_keys_argv[-1] == "Enter"
    assert f'eval "$(cat {command_path})"' in injected_line
    assert "large payload" not in injected_line
    assert len(injected_line) < 1_000
