import random
import torch
import logging
import multiprocessing
import numpy as np

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        choices=[
            "review",
        ],
    )
    parser.add_argument(
        "--model_type",
        default="codet5",
        type=str,
        choices=["roberta", "t5", "bart", "codet5", "scratch"],
    )
    parser.add_argument("--add_lang_ids", action="store_true")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--train_epochs", default=10, type=int)
    parser.add_argument("--tokenizer_path", type=str, required=False)
    
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--load_model_path",
        default=None,
        type=str,
        required=False
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Path to trained model: Should contain the .bin files",
    )
    ## Other parameters
    parser.add_argument(
        "--train_path",
        default=None,
        type=str,
        help="The pretrain files path. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--eval_chunkname",
        default=None,
        type=str,
        help="The eval file name.",
    )
    parser.add_argument(
        "--train_filename",
        default=None,
        type=str,
        help="The train filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--dev_filename",
        default=None,
        type=str,
        help="The dev filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--test_filename",
        default=None,
        type=str,
        help="The test filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--gold_filename",
        default=None,
        type=str,
        help="The gold filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--config_name",
        default="Salesforce/codet5-base",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_source_length",
        default=64,
        type=int,
        help="The maximum total source sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=32,
        type=int,
        help="The maximum total target sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run eval on the train set."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--raw_input", action="store_true", help="Whether to use simple input format (set for baselines)."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--mask_rate", default=0.15, type=float, help="The masked rate of input lines.",
    )
    parser.add_argument(
        "--beam_size", default=6, type=int, help="beam size for beam search"
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--save_steps", default=-1, type=int,
    )
    parser.add_argument(
        "--log_steps", default=-1, type=int,
    )
    parser.add_argument("--eval_steps", default=1000, type=int, help="")
    parser.add_argument("--eval_file", default="", type=str)
    parser.add_argument("--out_file", default="", type=str)
    parser.add_argument("--break_cnt", default=-1, type=int)
    parser.add_argument("--train_steps", default=-1, type=int, help="")
    parser.add_argument(
        "--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--gpu_per_node",
        type=int,
        default=4,
        help="gpus per node",
    )
    parser.add_argument(
        "--node_index",
        type=int,
        default=0,
        help="For distributed training: node_index",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--seed", type=int, default=2233, help="random seed for initialization"
    )  # previous one 42
    # Or in configs.py if add_args is defined there

    parser.add_argument(
        "--clearml_train_dataset_id",
        type=str,
        default=None,
        help="ClearML Dataset ID to fetch training data from. Overrides train_filename if provided.",
    )
    parser.add_argument(
        "--clearml_valid_dataset_id",
        type=str,
        default=None,
        help="ClearML Dataset ID to fetch validation data from. Overrides dev_filename if provided.",
    )
    parser.add_argument(
        "--clearml_load_task_id",
        type=str,
        default="6a5d6d1b47754c22a2f5b0b0d0e2c6f4",
        help="ClearML Task ID to load checkpoint from."
    )
    parser.add_argument(
        "--clearml_load_artifact_name",
        type=str,
        default="lora-adapter",
        help="Name of the ClearML artifact to load (e.g., 'lora-adapter' or 'checkpoints-1100')."
    )
    args = parser.parse_args()
    return args


def set_dist(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:
        # Setup for distributed data parallel
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    cpu_count = multiprocessing.cpu_count()
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        cpu_count,
    )
    args.device = device
    args.cpu_count = cpu_count


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)
