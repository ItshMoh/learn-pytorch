"""Lesson 10: padding is not a word, and an embedding is a learned lookup."""
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


def collate_tokens(sequences):
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    valid = torch.arange(padded.size(1))[None, :] < lengths[:, None]
    return padded, valid


def main():
    torch.manual_seed(3)
    sequences = [torch.tensor([1, 2, 3]), torch.tensor([2])]
    ids, valid = collate_tokens(sequences)
    embedding = nn.Embedding(8, 4, padding_idx=0)
    vectors = embedding(ids)
    pooled = (vectors * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True)
    print("IDs:\n", ids)
    print("valid tokens:\n", valid)
    print("vectors:", vectors.shape, "pooled:", pooled.shape)
    stream = torch.tensor([1, 2, 3, 4, 5])
    print("LM input:", stream[:-1], "target:", stream[1:])
    logits = nn.Linear(4, 8)(vectors)
    print("token logits:", logits.shape)


if __name__ == "__main__":
    main()
