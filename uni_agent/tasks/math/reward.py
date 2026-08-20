"""Math task reward computation."""

from __future__ import annotations

import re
from typing import Any


def _normalize_number(text: str) -> str:
    """Normalize a numeric string so ``27``, ``27.``, ``27.0`` all compare equal.

    The regex below can leave a trailing ``.`` (e.g. ``27.``) on the prediction,
    which would otherwise fail the string comparison against the ground truth.
    """
    text = text.strip()
    if not text:
        return text
    try:
        value = float(text)
    except ValueError:
        return text
    return str(int(value)) if value.is_integer() else str(value)


def compute_reward(sample: dict[str, Any], sandbox_output: str) -> dict[str, Any]:
    """Compare agent output against ground truth."""
    # sample is the task config metadata. Depending on how the dataset was built
    # the ground truth lives at the top level ({"ground_truth": ...}) or nested
    # under reward_model ({"reward_model": {"ground_truth": ...}}). Support both.
    ground_truth = sample.get("ground_truth") or sample.get("reward_model", {}).get("ground_truth", "")
    if not ground_truth:
        return {"resolved": 0, "score": 0, "ground_truth": ground_truth, "prediction": ""}

    # Extract the last number from the agent's output
    numbers = re.findall(r"-?\d+\.?\d*", sandbox_output)
    prediction = numbers[-1] if numbers else ""

    resolved = 1 if _normalize_number(prediction) == _normalize_number(str(ground_truth)) else 0
    return {"resolved": resolved, "score": resolved, "ground_truth": ground_truth, "prediction": prediction}
