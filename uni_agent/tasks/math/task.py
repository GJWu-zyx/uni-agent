"""Math task: simple arithmetic problems solved by an agent with tools."""

from __future__ import annotations

import logging

from pydantic import Field

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task

logger = logging.getLogger(__name__)

#: Guides even a tiny model to use the shell tool correctly (bare arithmetic like
#: ``12+15`` is not a valid command, so we tell it to compute via python).
_MATH_SYSTEM_PROMPT = (
    "You are an arithmetic solver. You have access to a `shell` tool that runs real commands.\n"
    "To compute an expression, always call the shell tool with a python one-liner, e.g.:\n"
    '{"command": "python3 -c \\"print(12+15)\\""}\n'
    "After you see the tool output, reply with ONLY the numeric answer (no explanation)."
)


def _final_answer_text(transcript: list[dict]) -> str:
    """Last plain-text assistant message (no tool calls) is the agent's answer."""
    for msg in reversed(transcript):
        if (
            msg.get("role") == "assistant"
            and not msg.get("tool_calls")
            and isinstance(msg.get("content"), str)
            and msg["content"].strip()
        ):
            return msg["content"]
    return ""


class MathTaskConfig(TaskConfig):
    name: str = "math"
    ground_truth: str = Field(default="", description="Expected answer.")


@register_task("math")
class MathTask(Task):
    name = "math"
    config_model = MathTaskConfig

    async def run(self) -> TaskResult:
        cfg: MathTaskConfig = self.config
        sample = cfg.metadata if isinstance(cfg.metadata, dict) else {}

        logger.info("starting math task: %s (ground_truth=%s)", sample.get("prompt", "?")[:80], cfg.ground_truth)

        async with self.build_sandbox() as sandbox:
            agent = self.build_agent()
            messages = list(cfg.prompt)
            # Inject a system prompt that teaches even a small model to compute via
            # ``python3 -c`` -- bare arithmetic (``12+15``) is not a valid shell command.
            if not any(m.get("role") == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": _MATH_SYSTEM_PROMPT})
            agent_result = await agent.run(sandbox=sandbox, messages=messages)

            # The policy answers arithmetic directly in plain text, so take the final
            # assistant reply from the transcript. Fall back to /tmp/math_answer.txt
            # (read from the sandbox) for agents that write their answer to a file.
            answer_text = _final_answer_text(agent_result.transcript)
            if not answer_text:
                try:
                    output = await sandbox.read_file("/tmp/math_answer.txt")
                    answer_text = output.decode("utf-8") if isinstance(output, bytes) else str(output)
                except Exception:
                    answer_text = ""

            from .reward import compute_reward

            # DEBUG rollout content: dump the full transcript so we can see what the
            # policy actually produced (role / content / tool_calls) when rewards look wrong.
            for i, msg in enumerate(agent_result.transcript):
                role = msg.get("role", "?")
                content = msg.get("content", "")
                tc = msg.get("tool_calls")
                if isinstance(content, str) and content.strip():
                    logger.info("transcript[%d] %s: %s", i, role.upper(), content[:400])
                else:
                    logger.info("transcript[%d] %s: (content=%r, tool_calls=%r)", i, role.upper(), content, tc)

            result = compute_reward(sample, answer_text)

            logger.info(
                "math task done: ground_truth=%r prediction=%r resolved=%s answer_text=%r",
                result["ground_truth"],
                result["prediction"],
                result["resolved"],
                answer_text,
            )
            return TaskResult(
                reward=float(result["resolved"]),
                accuracy=float(result["resolved"]),
                finished=agent_result.finished,
                extra_info=result,
            )
