"""Lesson 2: shape changes are not interchangeable."""
import torch


def main():
    x = torch.arange(24.0).reshape(2, 3, 4)  # B=2, T=3, C=4
    print("the declared x ",x )
    weights = torch.ones(4, 6)
    print("projection:", (x @ weights).shape)
    # @ is a matrix multiplication and * is for element wise multiplication.
    print("keep a reduction axis:", x.mean(dim=-1, keepdim=True).shape) # here we keepdim = True , it means the axes on which we calculate the mean will be there . new shpae will be (2,3,1)
    print("keep a reduction axis the mean is :", x.mean(dim=-1, keepdim=True))
    q = x.reshape(2, 3, 2, 2).transpose(1, 2)
    print("B,H,T,D:", q.shape, "contiguous:", q.is_contiguous())
    # now here contiguous means that the pytorch is reading the tensor as same as it is in memory layout
    merged = q.transpose(1, 2).reshape(2, 3, 4)
    print("shape of merged",merged.shape)
    torch.testing.assert_close(merged, x)
    # assert_close checks that the two tensors are same with tiny floating point differences. 
    # why is there a difference because computers can't store the most decimals exactly.
    matrix = torch.arange(6).reshape(2, 3)
    print("transpose rearranges axes:\n", matrix.transpose(0, 1))
    print("reshape regroups storage order:\n", matrix.reshape(3, 2))
    prediction = torch.tensor([[1.0], [2.0], [3.0]])
    target = torch.tensor([1.0, 2.0, 3.0])
    print("silent mistake:", (prediction - target).shape)
    print("intended:", (prediction.squeeze(-1) - target).shape)


if __name__ == "__main__":
    main()
