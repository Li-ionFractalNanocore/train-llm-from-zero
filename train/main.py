import os
import sys
from itertools import count
from dataclasses import dataclass, field
from typing import Optional

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import HfArgumentParser, AutoTokenizer, LlamaForCausalLM, LlamaConfig


@dataclass
class ModelArgs:
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    dropout: float = 0.1

    vocab_size: int = -1
    max_sequence_length: int = 256


@dataclass
class TrainArgs:
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-1
    device: str = 'cpu'

    eval_generate_steps: int = 100


@dataclass
class DataArgs:
    data_path: str = None
    tokenizer_path: str = None


class PretrainDataset(Dataset):
    def __init__(self, file, context_length=256):
        super().__init__()
        with open(file, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint16)
        data = data[:len(data) // context_length * context_length]
        self.data = data.reshape(-1, context_length)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        line = self.data[idx]
        x = np.array(line[:-1], dtype=np.int64)
        y = np.array(line[1:], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


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

    dataset = PretrainDataset(data_args.data_path, context_length=model_args.max_sequence_length)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    llama_config = LlamaConfig(
        vocab_size=model_args.vocab_size, hidden_size=model_args.d_model, num_hidden_layers=model_args.n_layers,
        num_attention_heads=model_args.n_heads, num_key_value_heads=model_args.n_kv_heads,
        intermediate_size=4 * model_args.d_model,
        eos_token_id=tokenizer.eos_token_id, attention_dropout=model_args.dropout,
    )

    llm = LlamaForCausalLM(llama_config).to(device=train_args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(llm.parameters(), lr=train_args.lr, weight_decay=train_args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))

    def train_epoch():
        progress_bar = tqdm(enumerate(dataloader), leave=False)
        for step, (x, y) in progress_bar:
            optimizer.zero_grad()
            x, y = x.to(train_args.device), y.to(train_args.device)
            out = llm(x).logits
            loss = criterion(out.view(-1, model_args.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix(loss=loss.item())

            if step % train_args.eval_generate_steps == 0:
                print(generate(llm, tokenizer, top_k=5))

    epochs = 1
    for epoch in range(epochs):
        train_epoch()


if __name__ == '__main__':
    main()
