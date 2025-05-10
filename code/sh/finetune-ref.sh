#!/usr/bin/env bash
mnt_dir="/home/codereview"

# Configuration for 2 GPU training
MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=23333 && echo MASTER_PORT: ${MASTER_PORT}
RANK=0 && echo RANK: ${RANK}
PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
NODES=1 && echo NODES: ${NODES}
NCCL_DEBUG=INFO

  # Install required packages if not already installed
pip install nltk tqdm transformers peft clearml -q

# Download NLTK punkt package
python -c "import nltk; nltk.download('punkt')"

# Make sure data directory exists
mkdir -p ../data

# Check if data files exist and their paths
echo "Checking data files..."
ls -la ../data


TRAIN_DATASET_ID="119168a8d2a54de889a4e0faced0eea7"
VALID_DATASET_ID="f03c1b21febe4c22ac498d38c511b216"

# --- ClearML Credentials ---
export CLEARML_API_ACCESS_KEY="0BY6NYDNBD6VGLUL1YZBXKI9TIB7X6"
export CLEARML_API_SECRET_KEY="1xh4AUJhtaX9TZpSLfZYfu3G4CZTW07CC8ryvaGeVwN__LnmfRd-wR333i6xQD-FhuE"
export CLEARML_API_SERVER="https://api.clear.ml"
export CLEARML_WEB_SERVER="https://app.clear.ml"
export CLEARML_FILES_SERVER="https://files.clear.ml"
# -------------------------

# NCCL configuration for better multi-GPU communication
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Try with a smaller batch size to avoid GPU memory issues
torchrun --nproc_per_node=${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../runfinetune.py  \
  --train_epochs 1 \
  --model_name_or_path microsoft/codereviewer \
  --output_dir ../../save/ref \
  --train_filename ../data/ref-train.jsonl \
  --dev_filename ../data/ref-valid.jsonl \
  --clearml_train_dataset_id "119168a8d2a54de889a4e0faced0eea7" \
  --clearml_valid_dataset_id "f03c1b21febe4c22ac498d38c511b216" \
  --max_source_length 200 \
  --max_target_length 200 \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 3e-4 \
  --gradient_accumulation_steps 1 \
  --mask_rate 0.15 \
  --save_steps 100 \
  --log_steps 100 \
  --train_steps 60000 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233 \
