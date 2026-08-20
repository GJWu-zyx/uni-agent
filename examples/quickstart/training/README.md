# Agentic RL single-node smoke test

Minimal, self-contained end-to-end test that validates the full **agentic RL**
training pipeline on **one node**:

```
rollout (ReAct + local sandbox + tools) -> reward -> policy update
```

This is a *smoke test*, not a production recipe: it intentionally uses a tiny
hand-made math dataset and a small model so a couple of RL steps already finish
and print rollout / reward / policy-update numbers. It complements the
task-specific MemAgent / DeepEyes recipes by validating the whole pipeline with
minimal setup.

## Files

| File | Purpose |
| --- | --- |
| `run_agentic_smoke.sh` | Launches the ppo trainer with the uni-agent agent-framework rollout on one node (CUDA or NPU). |
| `gen_dummy_math_data.py` | Generates `train.parquet` / `val.parquet` with a handful of simple arithmetic problems and their ground truth. |
| `task_config_math.yaml` | Task definition: `react` (white-box uni-agent agent) + `local` sandbox + shell/editor tools. |

## 0. Prerequisites

- A python env with `verl` and `uni_agent` installed (see the repo install docs).
- A small HF model that supports the tool-call format in `TOOL_PARSER`.
  Default: `Qwen3-1.7B` (Hermes `<tool_call>` format). Point `MODEL_PATH` at
  whatever you have locally — e.g. `Qwen2.5-1.5B-Instruct`.
- **NPU only**: a sourced CANN/Ascend environment (set `CANN_SET_ENV` /
  `ATB_SET_ENV` if your install is not under `~/miniconda3/envs/uni-agent`).

## 1. Generate the dummy dataset (optional)

The smoke test auto-generates the dataset when the parquet files are missing,
but you can also create it explicitly:

```bash
python examples/quickstart/training/gen_dummy_math_data.py \
    --data_dir "$HOME/verl/data" \
    --num_train 16 --num_val 4
```

Output: `train.parquet` (16 rows) and `val.parquet` (4 rows).

## 2. Run the smoke test

Launch from the repository root so both `verl/` and `uni_agent/` are importable.

**NPU (Ascend):**

```bash
DEVICE=npu NGPUS_PER_NODE=8 \
    bash examples/quickstart/training/run_agentic_smoke.sh
```

**CUDA:**

```bash
DEVICE=cuda NGPUS_PER_NODE=1 MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct \
    bash examples/quickstart/training/run_agentic_smoke.sh
```

Optional knobs (all overridable via env): `NNODES`, `NGPUS_PER_NODE`,
`MODEL_PATH`, `MODEL_NAME`, `TOOL_PARSER`, `TASK_CONFIG`, `TRAIN_BATCH_SIZE`,
`N_RESP`, `TOTAL_EPOCHS`, `RAY_ADDR`, `DATA_DIR`, `CKPTS_DIR`, `AGENT_LOG_DIR`.

## 3. What to look for

The config uses console logging only (no wandb). After a couple of steps you
should see the per-step metrics printed by `TaskRunnerV1`, including:

- **Reward** — `critic/rewards/mean` (and `critic/score/mean`) showing a
  non-zero value, since the dummy problems are trivially solvable.
- **Policy update** — `actor/loss`, `actor/pg_clipfrac`, `actor/grad_norm`,
  advancing `training/global_step`.
- **Rollout shape** — `response_length/mean`, `training/num_turns/mean` (the
  agent ran `num_turns` tool-call rounds per episode).

The rollout **success / failed** counts are printed per agent episode by the
task runner: each session's transcript is dumped to the per-step `task.log`
under `AGENT_LOG_DIR` (look for `math task done:` lines with the prediction
and whether it matched the ground truth).

## 4. Troubleshooting

- **Empty / zero reward**: the ground truth must reach the reward step. The
  dummy dataset carries it in the `reward_model` / `extra_info` fields; if
  `critic/rewards/mean` stays `0.0`, inspect a session's `task.log` to see what
  the agent actually produced and whether parsing matched.
- **`MODELING_BACKEND=veomni` in logs despite not using it**: an inherited
  default from the NPU/CANN environment; ignore it unless you actually enabled
  veomni.
- **NPU env not found**: export `CANN_SET_ENV` / `ATB_SET_ENV` to your install.
- **Model download**: keep `MODEL_PATH` pointing at a locally available model to
  avoid HF network access during the smoke test.
