import json

import pytest
import torch

from uni_agent.gateway.session.codec import MessageCodec

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search docs",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


class _QwenTemplateTokenizer:
    eos_token_id = 100_000

    def __init__(self, name_or_path: str, *, requires_user: bool, thinking_prompt: bool):
        self.name_or_path = name_or_path
        self.requires_user = requires_user
        self.thinking_prompt = thinking_prompt

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        del tools, return_dict, kwargs
        if self.requires_user and not any(message.get("role") == "user" for message in messages):
            raise ValueError("No user query found in messages.")

        parts = []
        for index, message in enumerate(messages):
            role = message["role"]
            content = self._normalize_content(message.get("content", ""))
            if role == "tool":
                if index > 0 and messages[index - 1].get("role") != "tool":
                    parts.append("<|im_start|>user\n")
                parts.append(f"<tool_response>\n{content}\n</tool_response>\n")
                if index == len(messages) - 1 or messages[index + 1].get("role") != "tool":
                    parts.append("<|im_end|>\n")
                continue

            parts.append(f"<|im_start|>{role}\n{content}")
            if role == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call.get("function", tool_call)
                    parts.append(f"\n<tool_call>\n<function={function['name']}>\n")
                    for name, value in (function.get("arguments") or {}).items():
                        if isinstance(value, dict | list):
                            value = json.dumps(value)
                        parts.append(f"<parameter={name}>\n{value}\n</parameter>\n")
                    parts.append("</function>\n</tool_call>")
            parts.append("<|im_end|>\n")

        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
            if self.thinking_prompt:
                parts.append("<think>\n")
        rendered = "".join(parts)
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        marker = "<|im_end|>"
        cursor = 0
        while cursor < len(text):
            if text.startswith(marker, cursor):
                ids.append(self.eos_token_id)
                cursor += len(marker)
            else:
                ids.append(ord(text[cursor]))
                cursor += 1
        return ids

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(
            "<|im_end|>" if int(token_id) == self.eos_token_id else chr(int(token_id)) for token_id in token_ids
        )

    def convert_tokens_to_ids(self, token):
        return self.eos_token_id if token == "<|im_end|>" else 0

    @staticmethod
    def _normalize_content(content):
        if isinstance(content, list):
            return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        return "" if content is None else str(content)


class _QwenProcessor:
    class _ImageProcessor:
        patch_size = 16

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.image_processor = self._ImageProcessor()

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def __call__(self, *, text, **kwargs):
        del kwargs
        return {"input_ids": torch.tensor([self.tokenizer.encode(text[0])], dtype=torch.long)}


def _messages(arguments):
    user = {"role": "user", "content": "Task"}
    assistant = {
        "role": "assistant",
        "content": "Inspect.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": arguments},
            }
        ],
    }
    tool = {
        "role": "tool",
        "name": "search",
        "tool_call_id": "call_1",
        "content": "Observation:\nresult",
    }
    return user, assistant, tool


@pytest.mark.parametrize(
    ("model_name", "requires_user", "thinking_prompt"),
    [
        ("Qwen/Qwen3-Coder-30B-A3B-Instruct", False, False),
        ("Qwen/Qwen3.5-4B", True, True),
    ],
)
def test_continuous_token_merge_matches_full_qwen_template(
    model_name,
    requires_user,
    thinking_prompt,
):
    tokenizer = _QwenTemplateTokenizer(
        model_name,
        requires_user=requires_user,
        thinking_prompt=thinking_prompt,
    )
    codec = MessageCodec(tokenizer)
    user, assistant_api, tool = _messages('{"query":"docs"}')
    _, assistant_template, _ = _messages({"query": "docs"})

    template_prefix = [user, assistant_template]
    prefix_ids = tokenizer.apply_chat_template(
        template_prefix,
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=False,
    )
    expected_ids = tokenizer.apply_chat_template(
        [*template_prefix, tool],
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=True,
    )
    runtime_ids = prefix_ids[: -len(codec._turn_separator)]

    result, response_mask, response_logprobs = codec.merge_incremental(
        [user, assistant_api],
        [tool],
        runtime_ids,
        [1],
        [-0.2],
        tools=TOOLS,
    )

    assert result.token_ids == expected_ids
    assert result.inserted_token_ids == tokenizer.encode("\n")
    appended_count = len(result.inserted_token_ids) + result.appended_token_count
    assert response_mask == [1] + [0] * appended_count
    assert response_logprobs == [-0.2] + [0.0] * appended_count
    appended_text = tokenizer.decode(result.token_ids[len(runtime_ids) :])
    assert appended_text.startswith("\n<|im_start|>user\n<tool_response>\nObservation:\nresult")


@pytest.mark.parametrize(
    ("model_name", "requires_user", "thinking_prompt"),
    [
        ("Qwen/Qwen3-Coder-30B-A3B-Instruct", False, False),
        ("Qwen/Qwen3.5-4B", True, True),
    ],
)
def test_continuous_token_user_merge_preserves_runtime_and_appends_qwen_turn(
    model_name,
    requires_user,
    thinking_prompt,
):
    tokenizer = _QwenTemplateTokenizer(
        model_name,
        requires_user=requires_user,
        thinking_prompt=thinking_prompt,
    )
    codec = MessageCodec(tokenizer)
    previous_messages = [
        {"role": "user", "content": "Task"},
        {"role": "assistant", "content": "Previous response."},
    ]
    retry_message = {"role": "user", "content": "No tool call found. Please retry."}
    runtime_ids = tokenizer.apply_chat_template(
        [previous_messages[0]],
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=True,
    )
    assistant_completion = (
        "\n</think>\n\nPrevious response.<|im_end|>" if thinking_prompt else "Previous response.<|im_end|>"
    )
    runtime_ids += tokenizer.encode(assistant_completion, add_special_tokens=False)
    expected_append = "\n<|im_start|>user\nNo tool call found. Please retry.<|im_end|>\n<|im_start|>assistant\n" + (
        "<think>\n" if thinking_prompt else ""
    )

    result, response_mask, response_logprobs = codec.merge_incremental(
        previous_messages,
        [retry_message],
        runtime_ids,
        [1],
        [-0.2],
        tools=TOOLS,
    )

    assert result.token_ids == runtime_ids + tokenizer.encode(expected_append, add_special_tokens=False)
    appended_count = len(result.inserted_token_ids) + result.appended_token_count
    assert response_mask == [1] + [0] * appended_count
    assert response_logprobs == [-0.2] + [0.0] * appended_count
    appended_text = tokenizer.decode(result.token_ids[len(runtime_ids) :])
    assert appended_text == expected_append


def test_qwen35_processor_fallback_also_matches_full_template():
    tokenizer = _QwenTemplateTokenizer(
        "Qwen/Qwen3.5-4B",
        requires_user=True,
        thinking_prompt=True,
    )
    codec = MessageCodec(tokenizer, processor=_QwenProcessor(tokenizer))
    user, assistant, tool = _messages({"query": "docs"})
    previous_messages = [user, assistant]
    prefix_ids = tokenizer.apply_chat_template(
        previous_messages,
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=False,
    )
    expected_ids = tokenizer.apply_chat_template(
        [*previous_messages, tool],
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=True,
    )
    runtime_ids = prefix_ids[: -len(codec._turn_separator)]

    result, response_mask, response_logprobs = codec.merge_incremental(
        previous_messages,
        [tool],
        runtime_ids,
        [1],
        tools=TOOLS,
    )

    assert codec._continuous_token_builder is None
    assert result.token_ids == expected_ids
    assert response_mask == [1] + [0] * result.appended_token_count
    assert response_logprobs is None
