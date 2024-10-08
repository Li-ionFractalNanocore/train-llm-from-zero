import math
import os
import sys
import time
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

from tqdm import tqdm
import wandb

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, AutoTokenizer, LlamaForCausalLM, LlamaConfig, get_cosine_schedule_with_warmup
import transformers.utils
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from train.data.pretrain_dataset import PretrainDataset, MultiPretrainDataset


@dataclass
class ModelArgs:
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    dropout: float = 0.1

    vocab_size: int = -1
    max_sequence_length: int = 256

    lr: float = 5e-4
    weight_decay: float = 1e-1
    num_cycles: float = 10.0

    batch_size: int = 16
    grad_accum: int = None


@dataclass
class TrainArgs:
    device: str = 'cpu'
    mixed_precision: str = None
    compile: bool = False
    fused: bool = False

    max_steps: int = 0
    eval_steps: int = 1000
    checkpoint_save_steps: int = 1000
    resume: bool = False


@dataclass
class DataArgs:
    train_file_path: str = field(metadata={'action': 'append', 'type': str})
    project_name: str = 'llm-pretrain'
    seed: int = None
    probs: str = None
    valid_file_path: str = None
    test_file_path: str = None
    tokenizer_path: str = None
    project_dir: str = None


logger = get_logger('llm')


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, context_length):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length

        pe = torch.zeros(context_length, d_model)
        position = torch.arange(0, context_length).unsqueeze(1).float()
        div_term = torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1), :]


def generate(model, tokenizer, max_length=256, question='我们的目标是', temperature=1.0, top_k=None):
    input_ids = tokenizer.encode(question, return_tensors='pt').to(model.device)
    eos_id = tokenizer.eos_token_id

    model.eval()
    for i in range(max_length):
        with torch.no_grad():
            output: torch.Tensor = model(input_ids).logits
        logits = output[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_logits = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_logits, torch.tensor(float('-inf')).to(logits.device), logits)
        if temperature > 0:
            logits_div = logits / temperature
            probs = torch.softmax(logits_div, dim=-1)
            id_next = torch.multinomial(probs, num_samples=1)
        else:
            id_next = torch.argmax(logits, dim=-1, keepdim=True)
        if id_next.item() == eos_id:
            break
        input_ids = torch.cat([input_ids, id_next], dim=-1)
    return tokenizer.decode(input_ids[0])


def print_parameters(model):
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {num_params:,}", main_process_only=True)


def get_subfolders_sorted_by_time(folder: Path):
    if not folder.exists():
        return []
    return sorted([f for f in folder.iterdir() if f.is_dir()], key=lambda f: f.stat().st_ctime)


def load_checkpoint(accelerator: Accelerator, path: str):
    path = Path(path)
    checkpoints = get_subfolders_sorted_by_time(path)
    if checkpoints:
        accelerator.load_state(checkpoints[-1])


def save_checkpoint(accelerator: Accelerator, path: str, max_checkpoints: int = 5):
    path = Path(path)
    checkpoints = get_subfolders_sorted_by_time(path)
    if len(checkpoints) >= max_checkpoints:
        shutil.rmtree(checkpoints[0])
    save_path = Path(path) / f"checkpoint_{accelerator.step}"
    accelerator.save_state(save_path)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    parser = HfArgumentParser((ModelArgs, TrainArgs, DataArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        model_args, train_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith('.yaml'):
        model_args, train_args, data_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, train_args, data_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator(
        mixed_precision=train_args.mixed_precision,
        gradient_accumulation_steps=model_args.grad_accum,
        project_dir=data_args.project_dir,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    if data_args.seed is not None:
        set_seed(data_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_path, trust_remote_code=True)
    model_args.vocab_size = tokenizer.vocab_size

    probs = list(map(float, data_args.probs.split(',')))
    datasets = []
    for path in data_args.train_file_path:
        train_dataset = PretrainDataset(path, context_length=model_args.max_sequence_length, shift=False)
        datasets.append(train_dataset)
    train_dataset = MultiPretrainDataset(datasets, probs)
    total_batch_size = model_args.batch_size * accelerator.num_processes
    total_batch_tokens = total_batch_size * model_args.max_sequence_length
    logger.info(f'Dataset examples: {len(train_dataset)}')
    logger.info(f'Dataset tokens: {len(train_dataset) * model_args.max_sequence_length}')
    logger.info(
        f'Training total tokens: {train_args.max_steps * model_args.batch_size * model_args.max_sequence_length * accelerator.num_processes}')
    logger.info(f'Effective batch size: {total_batch_size}')
    logger.info(f'Effective tokens per batch: {total_batch_tokens}')
    logger.info(
        f'Real tokens per batch: {total_batch_tokens * accelerator.gradient_accumulation_steps * accelerator.num_processes}')
    train_dataloader = DataLoader(train_dataset, batch_size=model_args.batch_size, shuffle=True, drop_last=True)
    valid_dataset = PretrainDataset(data_args.valid_file_path, context_length=model_args.max_sequence_length,
                                    shift=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=model_args.batch_size, shuffle=False, drop_last=True)
    logger.info(f'Dataset total steps: {len(train_dataloader)}')

    max_steps = train_args.max_steps if train_args.max_steps > 0 else len(train_dataloader)
    scheduler_steps = max_steps if model_args.grad_accum is None else max_steps // model_args.grad_accum
    warmup_steps = 8000 * accelerator.gradient_accumulation_steps

    llama_config = LlamaConfig(
        vocab_size=model_args.vocab_size, hidden_size=model_args.d_model, num_hidden_layers=model_args.n_layers,
        num_attention_heads=model_args.n_heads, num_key_value_heads=model_args.n_kv_heads,
        intermediate_size=4 * model_args.d_model,
        eos_token_id=tokenizer.eos_token_id, attention_dropout=model_args.dropout,
    )

    llm = LlamaForCausalLM(llama_config)
    print_parameters(llm)
    if train_args.fused:
        from apex.optimizers import FusedAdam
        optimizer = FusedAdam(llm.parameters(), lr=model_args.lr, weight_decay=model_args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(llm.parameters(), lr=model_args.lr, weight_decay=model_args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, scheduler_steps,
                                                num_cycles=model_args.num_cycles)

    if train_args.compile:
        orig_llm = llm
        llm = torch.compile(llm)

    llm, optimizer, scheduler, train_dataloader = accelerator.prepare(
        llm, optimizer, scheduler, train_dataloader,
    )
    total_steps_per_device = len(train_dataloader)
    logger.info(f'Total steps per device per epoch: {total_steps_per_device}')
    epochs = math.ceil(max_steps / total_steps_per_device)
    logger.info(f'Total epochs: {epochs}')

    if train_args.resume and data_args.project_dir is not None and os.path.exists(data_args.project_dir):
        load_checkpoint(accelerator, data_args.project_dir)
        logger.info(f'Training at {accelerator.step}')
        starting_epoch = accelerator.step // total_steps_per_device
        resume_steps = accelerator.step % total_steps_per_device
    else:
        starting_epoch = 0
        resume_steps = 0

    if accelerator.is_main_process:
        wandb_logger = wandb.init(project=data_args.project_name, config=asdict(model_args).update(asdict(train_args)))

    progress_bar = tqdm(range(max_steps), leave=True, disable=not accelerator.is_main_process)
    progress_bar.update(accelerator.step)

    def eval_epoch(model, eval_data_steps=100):
        model.eval()
        valid_losses = []
        perplexities = []
        for steps, (x, y) in enumerate(valid_dataloader):
            x, y = x.to(train_args.device), y.to(train_args.device)
            with torch.no_grad():
                out = model(x, labels=y)
                loss = out.loss
                perplexity = torch.exp(loss)
                valid_losses.append(loss.item())
                perplexities.append(perplexity.item())
                if steps >= eval_data_steps:
                    break
        return np.mean(valid_losses), np.mean(perplexities)

    all_tokens = accelerator.step * model_args.batch_size * model_args.max_sequence_length
    for epoch in range(starting_epoch, epochs):
        train_losses = []

        if train_args.resume and epoch == starting_epoch and resume_steps > 0:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_steps)
        else:
            active_dataloader = train_dataloader

        begin_time = time.time()
        for step, (x, y) in enumerate(active_dataloader):
            if accelerator.step >= max_steps:
                break

            llm.train()
            with accelerator.accumulate(llm):
                optimizer.zero_grad()
                with accelerator.autocast():
                    out = llm(x, labels=y)
                loss = out.loss
                if accelerator:
                    accelerator.backward(loss)
                else:
                    loss.backward()
                optimizer.step()
                scheduler.step()

            progress_bar.update(1)
            if accelerator.sync_gradients:
                progress_bar.set_postfix(loss=loss.item(), all_tokens=all_tokens,
                                         lr=optimizer.param_groups[0]['lr'])

            all_tokens += x.size(0) * x.size(1)
            train_losses.append(loss.item())

            if accelerator.is_main_process:
                if accelerator.step % train_args.eval_steps == 0:
                    gen_text = generate(llm, tokenizer, max_length=100, top_k=5)
                    val_loss, perplexity = eval_epoch(llm)
                    now = time.time()
                    tokens_per_sec = model_args.batch_size * model_args.max_sequence_length * train_args.eval_steps / (
                            now - begin_time)
                    begin_time = now
                    wandb_logger.log({
                        'train_loss': np.mean(train_losses),
                        'val_loss': val_loss,
                        'perplexity': perplexity,
                        'lr': optimizer.param_groups[0]['lr'],
                        'tokens_per_sec': tokens_per_sec,
                    }, step=accelerator.step)
                    logger.info(gen_text)
                    train_losses.clear()

                if accelerator.step % train_args.checkpoint_save_steps == 0 and data_args.project_dir is not None:
                    save_checkpoint(accelerator, data_args.project_dir)


if __name__ == '__main__':
    main()
