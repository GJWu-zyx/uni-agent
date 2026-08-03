from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse


class ImageZoomInTool(BaseTool):
    """Stateful DeepEyes image crop tool used by the gateway example."""

    def __init__(self, config: dict, tool_schema):
        super().__init__(config=config, tool_schema=tool_schema)
        self._images: dict[str, Image.Image | None] = {}

    async def create(self, instance_id: str | None = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id, response = await super().create(instance_id=instance_id)
        create_kwargs = kwargs.get("create_kwargs") or {}
        image = create_kwargs.get("image", kwargs.get("image"))
        self._images[instance_id] = _coerce_image(image)
        return instance_id, response

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        del kwargs
        image = self._images.get(instance_id)
        if image is None:
            return ToolResponse(text="No source image is available for image_zoom_in_tool."), 0.0, {"error": "no_image"}

        try:
            bbox = _bbox_from_parameters(parameters, image.size)
        except ValueError as exc:
            return ToolResponse(text=f"Could not zoom in: {exc}"), 0.0, {"error": "invalid_bbox"}

        x1, y1, x2, y2 = bbox
        crop = image.crop((x1, y1, x2, y2))
        label = parameters.get("label")
        label_text = f" for {label}" if label else ""
        text = f"Zoomed in{label_text} at bbox [{x1}, {y1}, {x2}, {y2}]."
        return ToolResponse(text=text, image=[crop]), 0.0, {"bbox": bbox, "crop_size": crop.size}

    async def release(self, instance_id: str, **kwargs) -> None:
        del kwargs
        self._images.pop(instance_id, None)


def _coerce_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    raise ValueError("Unsupported image type for ImageZoomInTool")


def _bbox_from_parameters(parameters: dict[str, Any], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be a dict")

    bbox = parameters.get("bbox_2d")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("bbox_2d must be a list of 4 numbers")

    x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
    width, height = image_size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"bbox_2d must fit within image bounds {image_size}")
    return x1, y1, x2, y2
