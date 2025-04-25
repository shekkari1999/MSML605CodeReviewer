# batch size 6 for 16 GB GPU

  mnt_dir="/home/codereview"

  # Configuration for 2 GPU training
  MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
  MASTER_PORT=23333 && echo MASTER_PORT: ${MASTER_PORT}
  RANK=0 && echo RANK: ${RANK}
  PER_NODE_GPU=4 && echo PER_NODE_GPU: ${PER_NODE_GPU}
  WORLD_SIZE=4 && echo WORLD_SIZE: ${WORLD_SIZE}
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

  export CLEARML_API_ACCESS_KEY="IUZD0AET8S29Q5PCRQQVFHNEA33WF2"
  export CLEARML_API_SECRET_KEY="OKVNC_EnDpdk5wXOLlmru07Hw4Ik_pWAwx7Dl-WUbJ5i8EXnDzX-6ka_kuEDnI8kksI"
  export CLEARML_SERVER_HOST="https://app.clearml.com"
  
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
    --max_source_length 200 \
    --max_target_length 200 \
    --train_batch_size 64 \
    --learning_rate 3e-4 \
    --gradient_accumulation_steps 3 \
    --mask_rate 0.15 \
    --save_steps 1800 \
    --log_steps 100 \
    --train_steps 60000 \
    --gpu_per_node=${PER_NODE_GPU} \
    --node_index=${RANK} \
    --seed 2233
