# Code for ACL 2026 Submission

This repository provides the official implementation of the training framework and reward mechanisms described in our paper.

## 📂 Project Structure

```text
.
├── requirements.txt            # Python dependencies
├── scripts
│   ├── data_process.py         # Data preprocessing script (JSONL -> Parquet)
│   └── train_llama.sh          # Main training shell script (Ray + GRPO)
├── verl
│   └──verl
│      └── utils
│          └── reward_score
│              └── custom_reward.py # Custom reward function implementation
└── README.md
```

## 🛠️ Environment Setup

We recommend using Anaconda to manage the environment.

```bash
# 1. Create a new conda environment
conda create -n verl_env python=3.10
conda activate verl_env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the project in editable mode (if applicable)
pip install -e .
```

## 📊 Data Preparation

The training requires data in Parquet format. We provide a script `scripts/data_process.py` to convert standard JSONL data into the format required by the training script.

1. Open `scripts/data_process.py` and modify the input/output paths:
   ```python
   input_file = "/path/to/your/data/input_file.jsonl"  
   output_dir = "/path/to/your/output/directory"
   ```

2. Run the processing script:
   ```bash
   python scripts/data_process.py
   ```
   

## 🚀 Training

To start training:

1. Open `scripts/train_llama.sh` and update the  environment variables to match your local setup:

2. Run the training script:
   ```bash
   bash scripts/train_llama.sh
   ```


## ⚠️ Anonymous Note
This repository is anonymized for double-blind review. 