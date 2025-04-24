# batch size 6 for 16 GB GPU

  mnt_dir="/home/codereview"

  # Configuration for 2 GPU training
  MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
  MASTER_PORT=23333 && echo MASTER_PORT: ${MASTER_PORT}
  RANK=0 && echo RANK: ${RANK}
  PER_NODE_GPU=2 && echo PER_NODE_GPU: ${PER_NODE_GPU}
  WORLD_SIZE=2 && echo WORLD_SIZE: ${WORLD_SIZE}
  NODES=1 && echo NODES: ${NODES}
  NCCL_DEBUG=INFO

  # Install required packages if not already installed
  pip install nltk tqdm transformers peft -q
  
  # Download NLTK punkt package
  python -c "import nltk; nltk.download('punkt')"

  # Make sure data directory exists
  mkdir -p ../data
  
  # Check if data files exist and their paths
  echo "Checking data files..."
  ls -la ../data

  # Use torchrun instead of torch.distributed.launch with correct data paths
  torchrun --nproc_per_node=${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_finetune_ref.py  \
    --train_epochs 1 \
    --model_name_or_path microsoft/codereviewer \
    --output_dir ../../save/ref \
    --train_filename ../data/ref-train.jsonl \
    --dev_filename ../data/ref-valid.jsonl \
    --max_source_length 200 \
    --max_target_length 200 \
    --train_batch_size 6 \
    --learning_rate 3e-4 \
    --gradient_accumulation_steps 3 \
    --mask_rate 0.15 \
    --save_steps 1800 \
    --log_steps 100 \
    --train_steps 60000 \
    --gpu_per_node=${PER_NODE_GPU} \
    --node_index=${RANK} \
    --seed 2233
