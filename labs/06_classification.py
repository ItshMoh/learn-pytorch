"""Lesson 6: read common.py's training loop, then reproduce it yourself."""
import argparse
from torch.utils.data import DataLoader
from common import train_classifier, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model, _, history, (_, _, test) = train_classifier(device=args.device)
    print("first epoch:", history[0])
    print("last epoch:", history[-1])
    print("test (inspect once after choosing settings):",
          evaluate(model, DataLoader(test, batch_size=64), next(model.parameters()).device))


if __name__ == "__main__":
    main()
