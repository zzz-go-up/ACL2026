#!/bin/bash
set -x 


export ANACONDA_ENV_PATH="/opt/conda/envs/verl/bin"
export PROJECT_HOME="/path/to/your/project" 

ulimit -n 65536


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


echo "Stopping any existing Ray..."
ray stop --force

echo "Starting local Ray cluster with 8 GPUs..."
ray start --head --num-gpus=8

echo "Waiting for Ray to register GPUs..."
while true; do
    RAY_STATUS=$(ray status 2>/dev/null)
    if [ $? -ne 0 ]; then
        echo "Ray cluster is unavailable, exiting..."
        ray stop --force
        exit 1
    fi
    if echo "$RAY_STATUS" | grep -qE '/8\.0[[:space:]]+GPU'; then
        echo "All GPUs are available. Starting training..."
        break
    fi
    echo "Waiting for GPUs... Current Ray Status:"
    echo "$RAY_STATUS" | grep "GPU"
    sleep 5
done


wandb offline
export WANDB_MODE=offline 
HOME=$PROJECT_HOME



MODEL_STUDENT_PATH="meta-llama/Llama-3.1-4B" 
MODEL_TEACHER_PATH="meta-llama/Llama-3.3-70B-Instruct" 


TRAIN_PATH="/path/to/train.parquet"
TEST_PATH="/path/to/test.parquet"

SAVE_PATH=$HOME/checkpoints/llama-exp
mkdir -p ${SAVE_PATH}
export TENSORBOARD_DIR=${SAVE_PATH}/tensorboard_logs
mkdir -p $TENSORBOARD_DIR

# --- 4. Start GRPO Training ---
python3 -um verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_PATH \
    data.val_files=$TEST_PATH \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_STUDENT_PATH} \
    +actor_rollout_ref.ref.model.path=${MODEL_TEACHER_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=81920 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.2 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.checkpoint.save_contents=["model"] \
    algorithm.kl_ctrl.kl_coef=0.2 \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.model.target_modules=all-linear \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="llama-exp" \
    trainer.experiment_name="exp_name" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    +trainer.tensorboard_dir=$TENSORBOARD_DIR \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    custom_reward_function.path=$HOME/verl/verl/utils/reward_score/custom_reward.py \
    custom_reward_function.name=compute_score_R_acc_lc_cta_batch_async_chunk \
    reward_model.reward_manager=batch


echo "Training completed, stopping Ray..."
ray stop --force
