# https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html

# What is PyTorch?
# It’s a Python-based scientific computing package targeted at two sets of audiences:

#     A replacement for NumPy to use the power of GPUs
#     a deep learning research platform that provides maximum flexibility and speed

# 1. Getting Started

# 1.1 Tensors
# Tensors are similar to NumPy’s ndarrays, with the addition being that Tensors can also be used on a GPU to accelerate computing.

from __future__ import print_function
import torch

# NOTE
# An uninitialized matrix is declared, but does not contain
#  definite known values before it is used.
#  When an uninitialized matrix is created,
#  whatever values were in the allocated memory at the time will appear as the initial values.

# Construct a 5x3 matrix, uninitialized:

x = torch.empty(5, 3)
print("\nx = torch.empty(5, 3)")
print(x)

# Construct a randomly initialized matrix:

x = torch.rand(5, 3)
print("\nx = torch.rand(5, 3)")
print(x)

# Construct a matrix filled zeros and of dtype long:

x = torch.zeros(5, 3, dtype=torch.long)
print("\nx = torch.zeros(5, 3, dtype=torch.long)")
print(x)

# Construct a tensor directly from data:

x = torch.tensor([5.5, 3])
print("\nx = torch.tensor([5.5, 3])")
print(x)

# or create a tensor based on an existing tensor.
#  These methods will reuse properties of the input tensor,
#  e.g. dtype, unless new values are provided by user

x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print("\nx = x.new_ones(5, 3, dtype=torch.double)")
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print("\nx = torch.randn_like(x, dtype=torch.float)")
print(x)

# Get its size:

print("\nx.size()")
print(x.size())

# NOTE
# torch.Size is in fact a tuple, so it supports all tuple operations.


# 1.2 Operations
# There are multiple syntaxes for operations.
#  In the following example, we will take a look at the addition operation.

# Addition: syntax 1

y = torch.rand(5, 3)
print("\nx + y")
print(x + y)

# Addition: syntax 2

print("\ntorch.add(x, y)")
print(torch.add(x, y))

# Addition: providing an output tensor as argument

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print("\ntorch.add(x, y, out=result) / print(result)")
print(result)

# Addition: in-place

# adds x to y
y.add_(x)
print("\ny.add_(x)")
print(y)

# NOTE
# Any operation that mutates a tensor in-place is post-fixed with an _.
#  For example: x.copy_(y), x.t_(), will change x.

# You can use standard NumPy-like indexing with all bells and whistles!

print("\nx[:, 1]")
print(x[:, 1])

# Resizing: If you want to resize/reshape tensor, you can use torch.view:

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print("\nview() -> x.size(), y.size(), z.size()")
print(x.size(), y.size(), z.size())

# If you have a one element tensor, use .item() to get the value as a Python number

x = torch.randn(1)
print("\nx")
print(x)
print("\nx.item()")
print(x.item())

# Read later:

# 100+ Tensor operations, including transposing, indexing, slicing,
#  mathematical operations, linear algebra, random numbers, etc.,
#  are described [here](https://pytorch.org/docs/stable/torch.html).


# 2. NumPy Bridge

# 2.1 Converting a Torch Tensor to a NumPy array and vice versa is a breeze.
# 
# The Torch Tensor and NumPy array will share their underlying memory locations 
# (if the Torch Tensor is on CPU), and changing one will change the other.

# Converting a Torch Tensor to a NumPy Array

a = torch.ones(5)
print("\na = torch.ones(5)")
print(a)

# share their underlying memory locations 
b = a.numpy()
print("\nb = a.numpy()")
print(b)

# See how the numpy array changed in value.

a.add_(1)
print("\na.add_(1) / print(a)")
print(a)
print("\nb")
print(b)


# 2.2 Converting NumPy Array to Torch Tensor

# See how changing the np array changed the Torch Tensor automatically

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print("\na = np.ones(5)\nb = torch.from_numpy(a)\nnp.add(a, 1, out=a)")
print(a)
print(b)

# All the Tensors on the CPU except a CharTensor support converting to NumPy and back.


# 3. CUDA Tensors

# 需要 CUDA 才可以使用 GPU 計算，所以 torch.cuda.is_available() 判斷是否可以使用 GPU 進行運算
# CUDA（Compute Unified Device Architecture，统一计算架构）是由NVIDIA所推出的一種整合技術，
# 是該公司對於GPGPU的正式名稱。 透過這個技術，使用者可利用NVIDIA的GeForce 8以後的GPU和較新的Quadro GPU进行计算。
# Tensors can be moved onto any device using the .to method.

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

