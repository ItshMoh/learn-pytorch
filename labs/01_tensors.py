"""Lesson 1: predict each output before running."""
import torch


def main():
    x = torch.tensor([[1.0, 2.0, 3.0,10], [4.0, 5.0, 6.0,11],[7.0,8.0,9,12]])
    print("shape, dtype, device:", x.shape, x.dtype, x.device)
    print("first example:", x[0])
    print("second feature:", x[:, 1])
    print("feature means:", x.mean(dim=0))
    print("example means:", x.mean(dim=1))
    print("long IDs:", torch.tensor([0, 2, 1]).dtype)
    print("boolean selection:", x[x > 3])
    print("a tensor of zeros",torch.zeros(2,3))
    print("a tensor of zeros , here shape can be given in a sinlge tuple or as separated int also",torch.zeros((2,3)))
    print("the tensor of one",torch.ones(3)) # a single row with 3 columns
    print("a tensor filled with 5 with the help of torch.full ",torch.full((2,3),5.0)) # it fills the whole tensor with given shape with 5
    print("identity matrix",torch.eye(2,3))
    print("integer zero matrix",torch.zeros(2,3,dtype= torch.long)) 
    print("printing a values from a range",torch.arange(5)) # end is excluded same like python range
    print("printing from a range and step",torch.arange(2,10,3)) # end is excluded 
    print("a range of float values",torch.arange(0,1,0.25)) 
    print("printing a fixed number of points between a range with the end included",torch.linspace(0,1,5)) # 5 is the number of points required
    print("making random results more precise",torch.manual_seed(0))
    print("selecting a random integers ",torch.rand(2,5))
    print("selecting a random sample from normal distribution",torch.randn(2,5)) 
    print("selecting integere from a range with a definiet shape",torch.randint(0,10,(2,3))) # shape must be in tuple 
    print("random permutations of the numbers in a range",torch.randperm(5))
    out = torch.empty(2,3)
    print("out",out)
    x = torch.randn(4,3,dtype = torch.float64)
    x1 = torch.zeros_like(x)
    print("zeros like , it just copies the dtype and the shape of the original one",x1)
    # Conversion returns a tensor; it does not change x's dtype in place.
    y = x.to(dtype=torch.float64)
    print("original / converted:", x.dtype, y.dtype)


if __name__ == "__main__":
    main()
