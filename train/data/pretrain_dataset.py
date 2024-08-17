import glob
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, path, context_length=256, mmap=True):
        super().__init__()
        files = glob.glob(path)
        self.data = []
        self.indices = []
        for file in files:
            if mmap:
                with open(file, 'r') as f:
                    f.seek(0, 2)
                    length = f.tell() // np.dtype(np.uint16).itemsize
                index = length // context_length
                self.data.append(np.memmap(file, dtype=np.uint16, mode='r', shape=(index, context_length)))
                self.indices.append(index)
            else:
                with open(file, 'rb') as f:
                    data = np.fromfile(f, dtype=np.uint16)
                index = len(data) // context_length
                data = data[:index * context_length].reshape(-1, context_length)
                self.data.append(data)
        self.whole_length = sum(self.indices)
        self.indices = np.cumsum(self.indices)

    def __len__(self):
        return self.whole_length

    def __getitem__(self, idx):
        idx = idx % self.whole_length
        dataset_idx = np.searchsorted(self.indices, idx)
        line = self.data[dataset_idx][idx - self.indices[dataset_idx]]
        x = np.array(line[:-1], dtype=np.int64)
        y = np.array(line[1:], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


class MultiPretrainDataset(Dataset):
    def __init__(self, datasets, probabilities):
        assert len(datasets) == len(probabilities), "Number of datasets and probabilities must match"
        assert sum(probabilities) == 1, "Probabilities must sum to 1"

        self.datasets = datasets
        if sum(probabilities) != 1:
            raise ValueError("Probabilities must sum to 1")
        self.count = list(map(lambda x: int(x * 100), probabilities))
        self.count_accum = np.cumsum(self.count)
        self.count_accum = np.insert(self.count_accum, 0, 0)
        self.total_length = sum(len(dataset) for dataset in datasets)
        self.index_map = self._create_index_map()

    def _create_index_map(self):
        index_map = {}
        for i in range(100):
            dataset_index = np.searchsorted(self.count_accum, i, side='right')
            index_map[i] = dataset_index - 1
        return index_map

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # put 100 numbers into a circle to split the datasets
        dataset_idx = self.index_map[idx % 100]
        chosen_dataset = self.datasets[dataset_idx]
        inner_idx = idx // 100 * self.count[dataset_idx] + idx % 100 - self.count_accum[dataset_idx]
        return chosen_dataset[inner_idx]
