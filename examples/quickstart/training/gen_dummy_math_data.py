# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate a minimal dummy math dataset for the Uni-Agent single-node smoke test.

This produces the exact RL-dataset schema consumed by verl's ppo trainer
(``prompt``, ``data_source``, ``reward_model``, ``extra_info``), with the
ground truth stored inside ``reward_model.ground_truth`` so the rule-based
math reward can compare against it without any external judge.

The serialized Task Config for each sample rides in ``extra_info.tools_kwargs``
(``{"task": {"name": "math"}}``), matching what verl's ``rl_dataset``
extracts and what ``uni_agent.framework.task_runner.run_task`` consumes.

Run::

    python gen_dummy_math_data.py --data_dir ./data

Outputs ``./data/train.parquet`` and ``./data/val.parquet``.
"""

from __future__ import annotations

import argparse
import os
import random

import pandas as pd

# A small, deterministic pool of multi-step arithmetic expressions together
# with their exact integer answers. Unlike single-step ``3*7`` these are hard
# for a tiny 0.5B model to compute by head -- it must actually call the shell
# tool (``python3 -c "print(...)"``) -- so the reward starts low and the RL
# curve visibly rises as the model learns to use the tool. Every answer is a
# single integer so the rule-based math reward keeps working unchanged.
PROBLEMS: list[tuple[str, str]] = [
    ("Solve: (3*7-5)/2=?", "8"),
    ("Solve: 2^3+5*2-3=?", "15"),
    ("Solve: sqrt(36)*2+4=?", "16"),
    ("Solve: 100/4+3*2=?", "31"),
    ("Solve: (15+9)/4=?", "6"),
    ("Solve: 5*5-8/2=?", "21"),
    ("Solve: 2^4-3*5=?", "1"),
    ("Solve: sqrt(49)+2^2=?", "11"),
    ("Solve: (20%3)*10+5=?", "25"),
    ("Solve: 7+8*3-11=?", "20"),
    ("Solve: (6+2)^2/16=?", "4"),
    ("Solve: sqrt(144)-3*4=?", "0"),
    ("Solve: 17%5*3+4=?", "10"),
    ("Solve: (4*6)/3+9=?", "17"),
    ("Solve: 9*9/3-5=?", "22"),
    ("Solve: 2^5-5^2=?", "7"),
    ("Solve: (14-8)*5-2=?", "28"),
    ("Solve: 36/6+5*4=?", "26"),
    ("Solve: (12+15)/3=?", "9"),
    ("Solve: sqrt(81)+sqrt(16)=?", "13"),
]


def generate_problems(num: int, seed: int) -> list[tuple[str, str]]:
    """Return ``num`` problems; cycles the pool if ``num`` exceeds its size."""
    rng = random.Random(seed)
    return rng.sample(PROBLEMS, k=min(num, len(PROBLEMS)))


def build_dataset(problems: list[tuple[str, str]]) -> dict[str, list]:
    dataset: dict[str, list] = {
        "prompt": [],
        "data_source": [],
        "ability": [],
        "reward_model": [],
        "extra_info": [],
    }
    for prompt, answer in problems:
        # Per-sample serialized Task Config: prompt feeds the agent's first
        # chat/completions messages, metadata carries the ground truth used by
        # the math reward. Without it run_task sends an empty message list.
        task_config = {
            "name": "math",
            "prompt": [{"role": "user", "content": prompt}],
            "ground_truth": answer,
            "metadata": {"prompt": prompt, "ground_truth": answer},
        }
        dataset["prompt"].append(prompt)
        dataset["data_source"].append("math")
        dataset["ability"].append("math")
        dataset["reward_model"].append({"style": "rule", "ground_truth": answer})
        # extra_info.tools_kwargs carries the serialized Task Config per sample;
        # without it the framework's run_task has no task to resolve.
        dataset["extra_info"].append({"answer": answer, "tools_kwargs": {"task": task_config}})
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, default="./data", help="Output directory for the parquet files.")
    parser.add_argument("--num_train", type=int, default=16, help="Number of training samples.")
    parser.add_argument("--num_val", type=int, default=4, help="Number of validation samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    train = build_dataset(generate_problems(args.num_train, args.seed))
    val = build_dataset(generate_problems(args.num_val, args.seed + 1))

    train_path = os.path.join(args.data_dir, "train.parquet")
    val_path = os.path.join(args.data_dir, "val.parquet")
    pd.DataFrame(train).to_parquet(train_path)
    pd.DataFrame(val).to_parquet(val_path)

    print(f"Wrote {len(train['prompt'])} training samples -> {train_path}")
    print(f"Wrote {len(val['prompt'])} validation samples -> {val_path}")


if __name__ == "__main__":
    main()
