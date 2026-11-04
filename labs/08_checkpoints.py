"""Lesson 8: a checkpoint includes model state and training state."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from common import Classifier, train_classifier


def step(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()


def roundtrip(path):
    model, optimizer, _, (train, val, _) = train_classifier(epochs=3)
    model.eval()
    with torch.no_grad():
        before = model(val.x)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "config": {"hidden": 24, "dropout": 0.1}, "completed_epochs": 3,
        "cpu_rng": torch.get_rng_state(),
    }, path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    restored = Classifier(**checkpoint["config"])
    restored.load_state_dict(checkpoint["model"])
    restored.eval()
    with torch.no_grad():
        after = restored(val.x)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.02)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    # Compare the SAME next batch and dropout random draws in both branches.
    torch.set_rng_state(checkpoint["cpu_rng"])
    step(model, optimizer, train.x[:32], train.y[:32])
    torch.set_rng_state(checkpoint["cpu_rng"])
    step(restored, resumed_optimizer, train.x[:32], train.y[:32])
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs/classifier.pt")
    args = parser.parse_args()
    print("Saved:", roundtrip(args.output))
    print("Exact evaluation roundtrip and next CPU optimizer step matched.")


if __name__ == "__main__":
    main()
