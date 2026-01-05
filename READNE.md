# Anonymous Reproduction Package

This repository provides an anonymized codebase for reproducing the experiments in the accompanying ACL submission (double-blind review).

## Structure

- `verl/`  
  Core training framework and utilities.

- `verl/scripts/data_process.py`  
  Converts an input `.jsonl` file (chat-style `messages`) into the custom Parquet format used by training.

- `train_llama.sh`  
  Example training script (GRPO via `verl.trainer.main_ppo`) using Ray + vLLM rollout and a custom reward function.

- `verl/requirements.txt`  
  Python dependencies.

## Environment Setup

> The project is typically run on a Linux server with CUDA GPUs. The training demo script assumes 8 GPUs and a local Ray cluster.

### 1) Create environment (example)

Use any environment manager you prefer. For example, with conda:

```bash
conda create -n verl python=3.10 -y
conda activate verl
pip install -r verl/requirements.txt
