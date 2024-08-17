import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

from tqdm import tqdm
import wandb

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, AutoTokenizer, LlamaForCausalLM, LlamaConfig

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

    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-1


@dataclass
class TrainArgs:
    device: str = 'cpu'

    eval_steps: int = 1000
    checkpoint_save_steps: int = 1000


@dataclass
class DataArgs:
    train_file_path: str = field(metadata={'action': 'append', 'type': str})
    probs: str = None
    valid_file_path: str = None
    test_file_path: str = None
    tokenizer_path: str = None
    ckpt_path: str = None


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
    print(f"Total parameters: {num_params:,}")


def main():
    parser = HfArgumentParser((ModelArgs, TrainArgs, DataArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        model_args, train_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith('.yaml'):
        model_args, train_args, data_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, train_args, data_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_path, trust_remote_code=True)
    model_args.vocab_size = tokenizer.vocab_size

    probs = list(map(float, data_args.probs.split(',')))
    datasets = []
    for path in data_args.train_file_path:
        train_dataset = PretrainDataset(path, context_length=model_args.max_sequence_length)
        datasets.append(train_dataset)
    train_dataset = MultiPretrainDataset(datasets, probs)
    print(f'Total tokens: {len(train_dataset) * model_args.max_sequence_length}')
    train_dataloader = DataLoader(train_dataset, batch_size=model_args.batch_size, shuffle=True, drop_last=True)
    valid_dataset = PretrainDataset(data_args.valid_file_path, context_length=model_args.max_sequence_length)
    valid_dataloader = DataLoader(valid_dataset, batch_size=model_args.batch_size, shuffle=False, drop_last=True)

    llama_config = LlamaConfig(
        vocab_size=model_args.vocab_size, hidden_size=model_args.d_model, num_hidden_layers=model_args.n_layers,
        num_attention_heads=model_args.n_heads, num_key_value_heads=model_args.n_kv_heads,
        intermediate_size=4 * model_args.d_model,
        eos_token_id=tokenizer.eos_token_id, attention_dropout=model_args.dropout,
    )

    llm = LlamaForCausalLM(llama_config).to(device=train_args.device)
    print_parameters(llm)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(llm.parameters(), lr=model_args.lr, weight_decay=model_args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader))

    if data_args.ckpt_path is not None:
        ckpt_path = Path(data_args.ckpt_path)
        ckpt_file = ckpt_path / 'checkpoint.pt'
        if ckpt_file.exists():
            checkpoint = torch.load(ckpt_file)
            llm.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

    all_tokens = 0
    wandb_logger = wandb.init(project='llm-pretrain', config=asdict(model_args))

    def train_epoch():
        nonlocal all_tokens

        train_losses = []
        progress_bar = tqdm(enumerate(train_dataloader), leave=True)
        for step, (x, y) in progress_bar:
            optimizer.zero_grad()
            x, y = x.to(train_args.device), y.to(train_args.device)
            out = llm(x)
            logits = out.logits
            loss = criterion(logits.view(-1, model_args.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            all_tokens += x.size(0) * x.size(1)
            progress_bar.set_postfix(loss=loss.item(), all_tokens=all_tokens, lr=optimizer.param_groups[0]['lr'])
            train_losses.append(loss.item())

            if step % train_args.eval_steps == 0:
                gen_text = generate(llm, tokenizer, max_length=100, top_k=5)
                val_loss, perplexity = eval_epoch(llm)
                wandb_logger.log({
                    'train_loss': np.mean(train_losses),
                    'val_loss': val_loss,
                    'perplexity': perplexity,
                    'lr': optimizer.param_groups[0]['lr'],
                }, step=step)
                print(gen_text)
                train_losses.clear()

            if step % train_args.checkpoint_save_steps == 0 and data_args.ckpt_path is not None:
                ckpt_path.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model': llm.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, ckpt_file)

    def eval_epoch(model, eval_data_steps=100):
        model.eval()
        valid_losses = []
        perplexities = []
        progress_bar = tqdm(enumerate(valid_dataloader), leave=False, total=eval_data_steps)
        for steps, (x, y) in progress_bar:
            x, y = x.to(train_args.device), y.to(train_args.device)
            with torch.no_grad():
                out = model(x)
                logits = out.logits
                loss = criterion(logits.view(-1, model_args.vocab_size), y.view(-1))
                perplexity = torch.exp(loss)
                valid_losses.append(loss.item())
                perplexities.append(perplexity.item())
                if steps >= eval_data_steps:
                    break
        return np.mean(valid_losses), np.mean(perplexities)

    epochs = 1
    for epoch in range(epochs):
        train_epoch()


if __name__ == '__main__':
    main()
