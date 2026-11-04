"""Lesson 5: examples, batches, and preprocessing."""
import torch
from torch.utils.data import DataLoader
from common import make_splits


def main():
    train, val, test = make_splits()
    mean = train.x.mean(dim=0)
    scale = train.x.std(dim=0, correction=0).clamp_min(1e-6)
    print("split sizes:", len(train), len(val), len(test))
    print("one example:", train[0])
    loader = DataLoader(train, batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(1), num_workers=0)
    print("batch sizes:", [len(y) for _, y in loader])
    print("normalized training mean:", ((train.x - mean) / scale).mean(dim=0))
    print("validation uses TRAIN statistics:", ((val.x - mean) / scale).mean(dim=0))


if __name__ == "__main__":
    main()
