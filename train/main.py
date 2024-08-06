import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import HfArgumentParser, AutoTokenizer


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


class LLM(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.token_embedding = nn.Embedding(model_args.vocab_size, model_args.d_model)
        self.position_embedding = PositionalEncoding(model_args.d_model, context_length=model_args.max_sequence_length)
        self.transformer = nn.Transformer(d_model=model_args.d_model, nhead=model_args.n_heads, num_encoder_layers=0,
                                          num_decoder_layers=model_args.n_layers, dropout=model_args.dropout)
        self.output = nn.Linear(model_args.d_model, model_args.vocab_size)
        self.token_embedding.weight = self.output.weight

    def forward(self, x, y):
        x = self.token_embedding(x) + self.position_embedding(x)
        out = self.transformer(x)
        return self.output(out)


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

    llm = LLM(model_args).to(device=train_args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(llm.parameters(), lr=train_args.lr, weight_decay=train_args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))

    def train_epoch():
        for step, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            x, y = x.to(train_args.device), y.to(train_args.device)
            out = llm(x)
            loss = criterion(out.view(-1, model_args.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f"Step {step}, Loss: {loss.item()}")

    epochs = 1
    for epoch in range(epochs):
        train_epoch()


if __name__ == '__main__':
    main()
