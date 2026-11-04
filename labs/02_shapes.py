"""Lesson 2: shape changes are not interchangeable."""
import torch


def main():
    x = torch.arange(24.0).reshape(2, 3, 4)  # B=2, T=3, C=4
    print("the declared x ",x )
    weights = torch.ones(4, 6)
    print("projection:", (x @ weights).shape)
    print("keep a reduction axis:", x.mean(dim=-1, keepdim=True).shape)
    print("keep a reduction axis the mean is :", x.mean(dim=-1, keepdim=True))
    q = x.reshape(2, 3, 2, 2).transpose(1, 2)
    print("B,H,T,D:", q.shape, "contiguous:", q.is_contiguous())
    merged = q.transpose(1, 2).reshape(2, 3, 4)
    torch.testing.assert_close(merged, x)
    matrix = torch.arange(6).reshape(2, 3)
    print("transpose rearranges axes:\n", matrix.transpose(0, 1))
    print("reshape regroups storage order:\n", matrix.reshape(3, 2))
    prediction = torch.tensor([[1.0], [2.0], [3.0]])
    target = torch.tensor([1.0, 2.0, 3.0])
    print("silent mistake:", (prediction - target).shape)
    print("intended:", (prediction.squeeze(-1) - target).shape)


if __name__ == "__main__":
    main()
