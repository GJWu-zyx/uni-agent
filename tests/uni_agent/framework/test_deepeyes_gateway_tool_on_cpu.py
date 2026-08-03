from __future__ import annotations

import asyncio
from pathlib import Path

from PIL import Image

from verl.tools.tool_registry import initialize_tools_from_config


DEEPEYES_DIR = Path(__file__).resolve().parents[3] / "examples" / "agent_train" / "deepeyes_gateway"


def test_deepeyes_image_zoom_tool_config_initializes_and_crops_image():
    tools = initialize_tools_from_config(DEEPEYES_DIR / "configs" / "image_zoom_in_tool_config.yaml")

    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "image_zoom_in_tool"

    image = Image.new("RGB", (10, 8), color="white")
    image.putpixel((3, 4), (255, 0, 0))

    async def run_tool():
        instance_id, _ = await tool.create(create_kwargs={"image": image})
        try:
            response, reward, metrics = await tool.execute(
                instance_id,
                parameters={"bbox_2d": [2, 3, 6, 7], "label": "target"},
            )
        finally:
            await tool.release(instance_id)
        return response, reward, metrics

    response, reward, metrics = asyncio.run(run_tool())

    assert reward == 0.0
    assert metrics["crop_size"] == (4, 4)
    assert response.image is not None
    assert response.image[0].size == (4, 4)
    assert response.image[0].getpixel((1, 1)) == (255, 0, 0)
    assert "target" in response.text


def test_deepeyes_run_script_overrides_agent_runner_tool_config_path():
    script = (DEEPEYES_DIR / "run_deepeyes_gateway_grpo.sh").read_text()

    assert "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.deepeyes.tool_config_path" in script
    assert "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.deepeyes.runner_kwargs.max_turns" in script
    assert "actor_rollout_ref.rollout.custom.agent_framework.tool_config_path" not in script
    assert "actor_rollout_ref.rollout.custom.agent_framework.agent_runner_kwargs" not in script
