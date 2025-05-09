#!/usr/bin/env python3
import argparse
import os
import torch
import tqdm
import json
import multiprocessing
from itertools import cycle
from configs import add_args, set_seed, set_dist
import logging
import torch.distributed as dist
from models import build_or_load_gen_model
import time
import datetime
from peft import LoraConfig as LoRAConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from utils import RefineDataset
from torch.utils.data.distributed import DistributedSampler
from evaluator.smooth_bleu import bleu_fromstr
from torch.utils.data import SequentialSampler, DataLoader
from clearml import Task, Dataset, OutputModel  # Add import at the top level
import re

#### sets some high level logging info
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


#### This method is entirely responsible for tokenizing data

def get_loader(data_file, args, tokenizer, pool, eval = False):
    def fn(features):
        return features
    ### we use global_rank here because in DDP, each GPU should see only a subset of data
    global_rank = args.global_rank
    dataset = RefineDataset(tokenizer, pool, args, data_file)
    data_len = len(dataset)

    #### printing the dataset size before sharding
    if global_rank == 0:
        logger.info(f"Data length: {data_len}.")
    if eval:
        #### No shuffling is done
        sampler = SequentialSampler(dataset)
    else:
        ## Shuffling is done
        sampler = DistributedSampler(dataset)
    
    # Initialize DataLoader with limited number of workers to avoid memory issues
    # Using fewer workers, 4-8 is typically sufficient rather than using all available CPU cores
    num_workers = min(4, args.cpu_count)
    
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size, 
                           num_workers=num_workers, collate_fn=fn)
    return dataset, sampler, dataloader
    
#### If the above function is called, it returns the tokenized dataset


### eval function compares predictions with gold references(ground truth)
def eval_bleu_epoch(args, eval_dataloader, model, tokenizer, valid_file):  # Added valid_file argument
    logger.info(f"  ***** Running bleu evaluation on {valid_file} *****")
    logger.info("  Batch size = %d", args.eval_batch_size)
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    pred_ids, ex_ids = [], []
    for step, examples in enumerate(eval_dataloader, 1):
        source_ids = torch.tensor(
            [ex.source_ids for ex in examples], dtype=torch.long
        ).to(args.local_rank)
        source_mask = source_ids.ne(tokenizer.pad_id)
        preds = model.generate(
                            input_ids=source_ids,
                            attention_mask=source_mask,
                            use_cache=True,
                            num_beams=args.beam_size,
                            early_stopping=True,
                            max_length=args.max_target_length)
        top_preds = list(preds.cpu().numpy())
        pred_ids.extend(top_preds)
    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
    golds = []
    with open(valid_file, "r") as f:
        for line in f:
            golds.append(json.loads(line)["new"])
    golds = golds[:len(pred_nls)]
    for i in range(len(golds)):
        pred_nls[i], golds[i] = RefineDataset.process_pred_gold(pred_nls[i], golds[i])
    with open(os.path.join(args.output_dir, "preds.txt"), "w", encoding="utf-8") as f:
        for pred in pred_nls:
            f.write(pred.strip() + "\n")
    with open(os.path.join(args.output_dir, "golds.txt"), "w", encoding="utf-8") as f:
        for gold in golds:
            f.write(gold.strip() + "\n")
    em = 0
    for pred, gold in zip(pred_nls, golds):
        if " ".join(pred.split()) == " ".join(gold.split()):
            em += 1
    em = em / len(golds)
    logger.warning(f"EM: {em}")
    bleu = bleu_fromstr(pred_nls, golds, rmstop=False)

     
    # Add ClearML logging for metrics (only from rank 0 to avoid duplicates)
    if args.global_rank == 0:
        task = Task.current_task()
        if task:
            # Add safety check for global_step
            iteration = getattr(args, 'global_step', 0)
            task.logger.report_scalar("evaluation", "BLEU", value=bleu, iteration=iteration)
            task.logger.report_scalar("evaluation", "Exact Match", value=em, iteration=iteration)
            
            # Log a few examples for inspection
            if len(pred_nls) > 0:
                # Log 3 random examples (or fewer if less available)
                import random
                indices = random.sample(range(len(pred_nls)), min(3, len(pred_nls)))
                examples_text = ""
                for idx in indices:
                    examples_text += f"Example {idx}:\n"
                    examples_text += f"Gold: {golds[idx]}\n"
                    examples_text += f"Pred: {pred_nls[idx]}\n\n"
                task.logger.report_text(examples_text, iteration=iteration)
    return bleu


def save_model(model, optimizer, scheduler, output_dir, config):
    # If model is a PEFT/LoRA model, use its built‑in saver
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
        logger.info("Saved LoRA adapter weights to %s", output_dir)
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        config.save_pretrained(output_dir)
        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        output_optimizer_file = os.path.join(output_dir, "optimizer.pt")
        torch.save(
            optimizer.state_dict(),
            output_optimizer_file,
            _use_new_zipfile_serialization=False,
        )
        output_scheduler_file = os.path.join(output_dir, "scheduler.pt")
        torch.save(
            scheduler.state_dict(),
            output_scheduler_file,
            _use_new_zipfile_serialization=False

        )
     # Upload LoRA adapter folder as a generic artifact
    task = Task.current_task()
    task.upload_artifact(
        name="lora-adapter",
        artifact_object=output_dir
     )
 
    # Register adapter under ClearML Models registry
    out_model = OutputModel(
        task=task,
        name="AICodeReviewer-LoRA",
        framework="pytorch"
    )
    out_model.update_weights(
        weights_filename=os.path.join(output_dir, "pytorch_model.bin")
    )

def get_data_path(args, dataset_type, default_path, clearml_dataset_id, expected_filename):
    """Determines the data path, fetching from ClearML if an ID is provided."""
    data_path = default_path
    fetch_success = True  # Assume success initially

    if clearml_dataset_id:
        if args.global_rank == 0:  # Only rank 0 downloads
            logger.info(f"Attempting to fetch {dataset_type} dataset from ClearML ID: {clearml_dataset_id}")
            try:
                dataset_obj = Dataset.get(dataset_id=clearml_dataset_id)
                dataset_folder = dataset_obj.get_local_copy()
                fetched_path = os.path.join(dataset_folder, expected_filename)

                if os.path.exists(fetched_path):
                    data_path = fetched_path
                    logger.info(f"Using {dataset_type} data from ClearML dataset: {data_path}")
                else:
                    logger.error(f"File '{expected_filename}' not found in ClearML {dataset_type} dataset folder: {dataset_folder}")
                    fetch_success = False  # Mark fetch as failed
            except Exception as e:
                logger.error(f"Failed to get {dataset_type} dataset from ClearML (ID: {clearml_dataset_id}): {e}")
                fetch_success = False  # Mark fetch as failed

            # If fetch failed, log fallback
            if not fetch_success:
                logger.warning(f"Falling back to default {dataset_type} path: {default_path}")
                data_path = default_path  # Ensure fallback path is set

        # Synchronize all processes: wait for rank 0
        dist.barrier()

        # Broadcast the determined path and success status from rank 0
        if args.world_size > 1:
            broadcast_list = [data_path, fetch_success]
            dist.broadcast_object_list(broadcast_list, src=0)
            data_path = broadcast_list[0]
            fetch_success = broadcast_list[1]  # Other ranks get the success status
            logger.info(f"[Rank {args.global_rank}] Received {dataset_type} data path: {data_path} (Fetch success: {fetch_success})")

    else:
        logger.info(f"Using {dataset_type} data from command line argument: {data_path}")

    return data_path

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint directory by step number."""
    checkpoint_dirs = []
    pattern = re.compile(r"checkpoints-(\d+)-")
    if not os.path.exists(output_dir):
        return None, 0
    for d in os.listdir(output_dir):
        m = pattern.match(d)
        if m:
            step = int(m.group(1))
            checkpoint_dirs.append((step, os.path.join(output_dir, d)))
    if not checkpoint_dirs:
        return None, 0
    checkpoint_dirs.sort()
    return checkpoint_dirs[-1][1], checkpoint_dirs[-1][0]

def main(args):


    ### our main Distributed Data Parallel task starts, with this method
    dist.init_process_group(backend = 'nccl')

    ### rank of GPUs within the same machine is local rank\

    ### if there are 2 nodes and 3 GPUs in each node, then 
    ### local_rank = 0, 1, 2, 0, 1, 2
    ### global_rank = 0, 1, 2, 3, 4, 5
    ### node_index = 0, 0, 0, 1, 1, 1
    ### world_size = 6
    local_rank = dist.get_rank() % args.gpu_per_node
    args.global_rank = local_rank + args.node_index * args.gpu_per_node
    args.local_rank = local_rank
    args.world_size = dist.get_world_size()

    ### just some logging info again as what all we set
    logger.warning("Process rank: %s, global rank: %s, world size: %s, bs: %s",
                   args.local_rank, args.global_rank, \
                   torch.distributed.get_world_size(), \
                   args.train_batch_size)
    
    # Initialize ClearML only on the main process
    task = None
    if args.global_rank == 0:
        try:
            from clearml import Task
            task = Task.init(project_name="AI Powered CodeReviewer", task_name="Distributed Training")
            # Log configuration parameters
            task.connect_configuration(vars(args))
            logger.info("ClearML initialized successfully on main process")
        except Exception as e:
            logger.warning(f"ClearML could not be initialized: {str(e)}")
            task = None
    
    # Add small delay to stagger process initialization
    time.sleep(args.global_rank * 0.5)
    
    #### we gotta let each process work with their own GPUs.
    torch.cuda.set_device(local_rank)
    # Set seed early for consistent initialization
    set_seed(args)

    ## To track how much time it took to load the model 
    start_time = time.time()

    config, model, tokenizer = build_or_load_gen_model(args)
    
    ### Till here You have the core components loaded into memory
    ### The model weights are initialized 
    ### (either from scratch or from a pre-trained checkpoint)
   

    ## end time
    end_time = time.time()

    #### Here we can include some clearML task which would track model 
    ### Architecture information

    if task:
        task.connect_configuration({
            'model_config': config.to_dict(),
            'model_type': args.model_type
                })

        ### log some hardware configurations

        # Log GPU/hardware info
        task.connect_configuration({
            "num_gpus_total": args.world_size,
            "num_nodes": args.world_size // args.gpu_per_node,
            "gpus_per_node": args.gpu_per_node
        })

        # After model initialization
        task.logger.report_scalar(
            "initialization", "model_load_time_seconds", 
            value=(end_time - start_time), iteration=0
        )

    #### Lets Add some LoRA parameters to our model

    lora_cfg = LoRAConfig(
        r = 64,
        lora_alpha = 128, 
        target_modules=["q", "k", "v", "o", "wi", "wo"],  # T5 attention & MLP mats
        lora_dropout = 0.05,
        bias = 'none',
        task_type = 'SEQ_2_SEQ_LM'

    )

    #### Here we are adding our LoRA model configs to original model.

    model = get_peft_model(model, lora_cfg)

    ### Add this to our ClearML , so that we can track it for different runs

    if task:
        # After applying LoRA
        task.connect_configuration({
            "lora_config": {
                "r": lora_cfg.r,
                "lora_alpha": lora_cfg.lora_alpha,
                "target_modules": lora_cfg.target_modules,
                "lora_dropout": lora_cfg.lora_dropout,
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "total_params": sum(p.numel() for p in model.parameters()),
                "trainable_percentage": 100 * sum(p.numel() for p in model.parameters() if p.requires_grad) / 
                                        sum(p.numel() for p in model.parameters())
            }
        })
    ## some logging info again
    logger.info("LoRA params: %d trainable / %d total",
                sum(p.numel() for p in model.parameters() if p.requires_grad),
                sum(p.numel() for p in model.parameters()))

    #### Now its time to wrap up our model in DDP

    # Add logging and barrier before DDP initialization
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[Rank {args.global_rank}] Before DDP: Total Params={total_params}, Trainable Params={trainable_params}")
    
    # Ensure all processes are synchronized before initializing DDP
    # Simple barrier without timeout (compatible with older PyTorch versions)
    #dist.barrier()

    model = DDP(model.cuda(), device_ids = [local_rank],
                 output_device = local_rank, find_unused_parameters = True)

    pool = multiprocessing.Pool(args.cpu_count)

    #### Only LoRA Adapters are trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_grouped_parameters = [{"params": trainable_params, "weight_decay": 0.0}]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    args.warmup_steps = int(args.train_steps * 0.05)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps,
    )

##### This is all the checkpoint loading at this point
    if os.path.exists("{}/checkpoints-last/optimizer.pt".format(args.output_dir)):
        optimizer.load_state_dict(
            torch.load(
                "{}/checkpoints-last/optimizer.pt".format(args.output_dir),
                map_location="cpu",
            )
        )
        # Move scheduler loading inside the if block to prevent errors if file doesn't exist
        if os.path.exists("{}/checkpoints-last/scheduler.pt".format(args.output_dir)):
            scheduler.load_state_dict(
                torch.load(
                    "{}/checkpoints-last/scheduler.pt".format(args.output_dir),
                    map_location="cpu",
                )
            )
            # Track checkpoint loading 
            if task and args.global_rank == 0:
                task.logger.report_text("Resuming from checkpoint: {}/checkpoints-last/".format(args.output_dir))
                
                # log the current training step if available
                if hasattr(scheduler, "last_epoch"):
                    task.logger.report_scalar("training", "resumed_from_step", 
                                            value=scheduler.last_epoch, iteration=0)
    
    ##### Till here model part is finished.######

    #### Lets look at the data part now #####

    ### This is used for a lot of stuff like training progress tracking
    ### checkpoint saving, learning_rate_scheduling, logging, early stopping

    global_step = 0 
    
    save_steps = args.save_steps

    # --- Determine Data Paths using Helper Function ---
    train_file = get_data_path(args, "training", args.train_filename,
                               args.clearml_train_dataset_id, "ref-train.jsonl")
    valid_file = get_data_path(args, "validation", args.dev_filename,
                               args.clearml_valid_dataset_id, "ref-valid.jsonl")
    # --- End of Data Path Determination ---

    #### here go back to the first method of this page
    data_tuple = get_loader(train_file, args, tokenizer, pool)        # WARNING: this is a iterator, to save memory
    _, _, train_dataloader = data_tuple

    data_tuple = get_loader(valid_file, args, tokenizer, pool, eval=True)
    _, _, valid_dataloader = data_tuple
     # Check for latest checkpoint
    latest_ckpt_dir, latest_step = find_latest_checkpoint(args.output_dir)
    resume_from_ckpt = False
    if latest_ckpt_dir:
        model_ckpt = os.path.join(latest_ckpt_dir, "pytorch_model.bin")
        optimizer_ckpt = os.path.join(latest_ckpt_dir, "optimizer.pt")
        scheduler_ckpt = os.path.join(latest_ckpt_dir, "scheduler.pt")
        if os.path.exists(model_ckpt):
            logger.info(f"Resuming model from checkpoint: {latest_ckpt_dir}")
            state_dict = torch.load(model_ckpt, map_location="cpu")
            model.module.load_state_dict(state_dict, strict=False)
            resume_from_ckpt = True
        if os.path.exists(optimizer_ckpt):
            optimizer.load_state_dict(torch.load(optimizer_ckpt, map_location="cpu"))
        if os.path.exists(scheduler_ckpt):
            scheduler.load_state_dict(torch.load(scheduler_ckpt, map_location="cpu"))

    global_step = latest_step if resume_from_ckpt else 0
    if resume_from_ckpt:
        logger.info(f"[Rank {args.global_rank}] Training will resume from step {global_step}")

    for epoch in range(1, args.train_epochs + 1):
        # set seed for reproducible data split
        save_seed = args.seed
        args.seed += epoch
        set_seed(args)
        args.seed = save_seed
        
        # Properly set the epoch in the distributed sampler
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)
            logger.info(f"[Rank {args.global_rank}] Setting sampler epoch to {epoch}")
        
        model.train()
        nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
        for step, examples in enumerate(train_dataloader, 1):
            if step == 1:
                ex = examples[0]
                logger.info(f"[Rank {args.global_rank}] batch size: {len(examples)}")
                #logger.info(f"[Rank {args.global_rank}] example source: {tokenizer.convert_ids_to_tokens(ex.source_ids)[:20]}...")
                #logger.info(f"[Rank {args.global_rank}] example target: {tokenizer.convert_ids_to_tokens(ex.target_ids)[:20]}...")
            
            # Log GPU memory usage on the first step
            if step == 1:
                free_mem, total_mem = torch.cuda.mem_get_info(local_rank)
                free_mb = free_mem / (1024 * 1024)
                total_mb = total_mem / (1024 * 1024)
                used_mb = total_mb - free_mb
                logger.info(f"[Rank {args.global_rank}] GPU memory: using {used_mb:.2f}MB / {total_mb:.2f}MB ({(used_mb/total_mb)*100:.2f}%)")
            
            source_ids = torch.tensor(
                [ex.source_ids for ex in examples], dtype=torch.long
            ).to(local_rank)
            target_ids = torch.tensor(
                [ex.target_ids for ex in examples], dtype=torch.long
            ).to(local_rank)
            source_mask = source_ids.ne(tokenizer.pad_id)
            target_mask = target_ids.ne(tokenizer.pad_id)

            loss = model(
                input_ids=source_ids,
                input_labels=None,
                decoder_input_ids=target_ids,
                attention_mask=source_mask,
                decoder_attention_mask=target_mask,
                encoder_loss=False
            )

            if args.gpu_per_node > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()

            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            if nb_tr_steps % args.gradient_accumulation_steps == 0:
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                if args.global_rank == 0 and global_step % args.log_steps == 0:
                    # Some Clear ML Adding
                    progress_percent = (global_step / args.train_steps) * 100
                    if task:
                        task.logger.report_scalar("training", "progress", value=progress_percent, iteration=global_step)
                        current_lr = scheduler.get_last_lr()[0]  # Get current learning rate
                        task.logger.report_scalar("training", "learning_rate", value=current_lr, iteration=global_step)
                    train_loss = round(
                        tr_loss * args.gradient_accumulation_steps / nb_tr_steps,
                        4,
                    )
                    # Add ClearML loss tracking
                    if task:
                        task.logger.report_scalar("training", "loss", value=train_loss, iteration=global_step)
                    logger.info(
                        "step {}/{}: Train loss {}".format(
                            global_step,
                            args.train_steps,
                            round(train_loss, 3),
                        )
                    )
            if global_step == args.train_steps:
                # Synchronize all processes before final evaluation
                dist.barrier()
                
                # Only rank 0 performs final evaluation and checkpoint saving
                if args.global_rank == 0:
                    # end training
                    output_dir = os.path.join(args.output_dir, "checkpoints-last")
                    save_model(model, optimizer, scheduler, output_dir, config)
                    bleu = eval_bleu_epoch(args, valid_dataloader, model, tokenizer, valid_file)
                    save_model(model, optimizer, scheduler, output_dir, config)
                    logger.info(f"Reach max steps {args.train_steps}.")
                    time.sleep(5)
                
                # All processes should return together
                dist.barrier()
                return
            
            if global_step <= latest_step:
                continue  # Skip steps already completed
            if args.global_rank == 0 and \
                    global_step % save_steps == 0 and \
                    nb_tr_steps % args.gradient_accumulation_steps == 0:
                # Only rank 0 performs evaluation and checkpoint saving
                logger.info(f"[Rank {args.global_rank}] Starting checkpoint evaluation at step {global_step}")
                output_dir = os.path.join(args.output_dir, "checkpoints-" + str(global_step))
                save_model(model, optimizer, scheduler, output_dir, config)
                bleu = eval_bleu_epoch(args, valid_dataloader, model, tokenizer, valid_file)
                logger.info(
                    f"[Rank {args.global_rank}] Checkpoint saved: {global_step}-step model and optimizer into {output_dir}"
                )
                # Signal that rank 0 is done with checkpoint saving
                torch.cuda.synchronize()  # Ensure GPU operations are complete
            
            # All processes should wait here to ensure checkpoints are saved properly
            if global_step % save_steps == 0 and nb_tr_steps % args.gradient_accumulation_steps == 0:
                # First barrier: ensures rank 0 completes saving before other ranks proceed
                dist.barrier()
                logger.info(f"[Rank {args.global_rank}] Passed checkpoint barrier at step {global_step}, continuing training")


if __name__ == '__main__':

    ### make our arguments ready here
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    ### this is useful for knowing how many cpu cores are available for parallel 
    ### tasks

    args.cpu_count = multiprocessing.cpu_count()

    ### This will suppress all the low level warnings from HuggingFace and show 
    ### us only critical level
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    ## This will log all the arguments passes(like hyper parameters)
    logger.info(args)

    main(args)

    logger.info('Training finished')
