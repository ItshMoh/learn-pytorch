"""Tiny educational character LM. No pretrained weights or downloaded corpus."""
import argparse
import math
import random
import string
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from attention import CausalSelfAttention
from common import pick_device

CHARS = string.ascii_lowercase + " .,;:\n"


class CharacterWindows(Dataset):
    def __init__(self, text, block_size):
        self.ids = torch.tensor([CHARS.index(char) for char in text], dtype=torch.long)
        self.block_size = block_size
        self.starts = range(0, len(self.ids) - block_size, block_size)
        if len(self.starts) == 0:
            raise ValueError("Text must contain more than block_size characters")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        chunk = self.ids[start:start + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def make_corpus():
    colors = ["red", "blue", "green", "gold", "white", "black", "pink", "gray"]
    objects = ["box", "bird", "boat", "leaf", "stone", "cup", "kite", "book"]
    # Unique source documents; split documents before constructing token windows.
    docs = [f"the {color} {obj} is here. i see the {color} {obj}.\n"
            for color in colors for obj in objects]
    random.Random(11).shuffle(docs)
    return "".join(docs[:48]), "".join(docs[48:])


class Block(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = CausalSelfAttention(channels, heads)
        self.norm2 = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, 4 * channels), nn.GELU(), nn.Linear(4 * channels, channels)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab_size=len(CHARS), block_size=32, channels=32, heads=4, layers=2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, channels)
        self.position_embedding = nn.Embedding(block_size, channels)
        self.blocks = nn.ModuleList([Block(channels, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(channels)
        self.output = nn.Linear(channels, vocab_size)

    def forward(self, ids):
        _, t = ids.shape
        if t > self.block_size:
            raise ValueError("Crop input to block_size before calling the model")
        positions = torch.arange(t, device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    y.reshape(-1), reduction="sum").item()
        tokens += y.numel()
    return loss_sum / tokens


@torch.no_grad()
def generate(model, prompt="the ", count=100, temperature=0.8):
    if not prompt or any(char not in CHARS for char in prompt):
        raise ValueError("Use a nonempty prompt containing only the course alphabet")
    if temperature <= 0:
        raise ValueError("temperature must be positive; use argmax for greedy decoding")
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor([[CHARS.index(char) for char in prompt]], device=device)
    for _ in range(count):
        logits = model(ids[:, -model.block_size:])[:, -1, :] / temperature
        next_id = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    return "".join(CHARS[index] for index in ids[0].tolist())


def train(steps=300, device="cpu", output=None):
    torch.manual_seed(23)
    device = pick_device(device)
    config = {"vocab_size": len(CHARS), "block_size": 32,
              "channels": 32, "heads": 4, "layers": 2}
    train_text, val_text = make_corpus()
    train_data = CharacterWindows(train_text, config["block_size"])
    val_data = CharacterWindows(val_text, config["block_size"])
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True,
                              generator=torch.Generator().manual_seed(24))
    val_loader = DataLoader(val_data, batch_size=16)
    model = TinyLM(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    before = evaluate(model, val_loader, device)
    batches = iter(train_loader)
    for _ in range(steps):
        try:
            x, y = next(batches)
        except StopIteration:
            batches = iter(train_loader)
            x, y = next(batches)
        model.train()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(CHARS)), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    after = evaluate(model, val_loader, device)
    metrics = {"validation_before": before, "validation_after": after,
               "perplexity_after": math.exp(after), "steps": steps,
               "parameters": sum(p.numel() for p in model.parameters())}
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": config,
                    "characters": CHARS, "metrics": metrics}, output)
    return model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs/tiny_lm.pt")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    torch.set_num_threads(1)  # Tiny CPU operations can be slower with many threads.
    model, metrics = train(args.steps, args.device, args.output)
    print(metrics)
    print("Saved:", args.output)
    print("Sample (a toy, not a chatbot):", generate(model))


if __name__ == "__main__":
    main()
