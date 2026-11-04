"""Shared helpers. Read these alongside Lessons 5 and 6; no hidden training API."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


def pick_device(name="cpu"):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class PointsDataset(Dataset):
    def __init__(self, x, y):
        if len(x) != len(y):
            raise ValueError("One label is required for each example")
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def make_splits():
    """Three clusters. Split examples BEFORE computing normalization statistics."""
    generator = torch.Generator().manual_seed(41)
    centers = torch.tensor([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])
    y = torch.arange(720) % 3
    x = centers[y] + 0.65 * torch.randn(720, 2, generator=generator)
    order = torch.randperm(len(y), generator=generator)
    train, val, test = order[:480], order[480:600], order[600:]
    return tuple(PointsDataset(x[i], y[i]) for i in (train, val, test))


class Classifier(nn.Module):
    def __init__(self, hidden=24, dropout=0.1):
        super().__init__()
        # These statistics are saved and moved with the model, but not optimized.
        self.register_buffer("mean", torch.zeros(2))
        self.register_buffer("scale", torch.ones(2))
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 3)
        )

    def forward(self, x):
        return self.net((x - self.mean) / self.scale)

    @torch.no_grad()
    def fit_preprocessing(self, train_x):
        self.mean.copy_(train_x.mean(dim=0))
        self.scale.copy_(train_x.std(dim=0, correction=0).clamp_min(1e-6))


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(y)
        count += len(y)
    return loss_sum / count


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        count += len(y)
    return {"loss": loss_sum / count, "accuracy": correct / count}


def train_classifier(epochs=20, device="cpu"):
    torch.manual_seed(7)
    device = pick_device(device)
    train, val, test = make_splits()
    model = Classifier().to(device)
    model.fit_preprocessing(train.x.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.01)
    loader = DataLoader(train, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(8))
    history = []
    for _ in range(epochs):
        train_loss = train_epoch(model, loader, optimizer, device)
        val_metrics = evaluate(model, DataLoader(val, batch_size=64), device)
        history.append({"train_loss": train_loss, **val_metrics})
    return model, optimizer, history, (train, val, test)
