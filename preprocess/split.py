import argparse
from pathlib import Path

import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='Split a file into multiple files.')
    parser.add_argument('input_file', type=str, help='Input file to split.')
    parser.add_argument('output_dir', type=str, help='Output directory to store the split files.')
    return parser.parse_args()


def train_test_split(data, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    return data[:train_size], data[train_size:]


def main():
    args = get_args()
    with open(args.input_file, 'r') as f:
        data = np.fromfile(f, dtype=np.uint16)
    train_data, test_data = train_test_split(data, 0.8)
    valid_data, test_data = train_test_split(test_data, 0.5)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data.tofile(output_dir / 'train.bin')
    valid_data.tofile(output_dir / 'valid.bin')
    test_data.tofile(output_dir / 'test.bin')


if __name__ == '__main__':
    main()
