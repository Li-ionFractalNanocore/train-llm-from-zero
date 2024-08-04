import numpy as np

import torch
import torch.nn as nn


class PretrainDataset(nn.Module):
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
        x, y = line[:-1], line[1:]
        return torch.tensor(x), torch.tensor(y)

