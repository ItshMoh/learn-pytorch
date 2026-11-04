"""Optional CUDA float16 training example; safely skips on CPU/MPS machines."""
import torch
from torch import nn
from torch.nn import functional as F


def main():
    if not torch.cuda.is_available():
        print("Skipped: this optional float16 AMP example needs CUDA. Core labs use float32.")
        return
    model = nn.Linear(8, 3).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(16, 8, device="cuda")
    y = torch.randint(0, 3, (16,), device="cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = F.cross_entropy(model(x), y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    print("AMP training step completed:", loss.item())


if __name__ == "__main__":
    main()
