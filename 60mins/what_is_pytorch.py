# What is PyTorch?
# It’s a Python-based scientific computing package targeted at two sets of audiences:

#     A replacement for NumPy to use the power of GPUs
#     a deep learning research platform that provides maximum flexibility and speed

# Getting Started

# Tensors
# Tensors are similar to NumPy’s ndarrays, with the addition being that Tensors can also be used on a GPU to accelerate computing.

from __future__ import print_function
import torch

# An uninitialized matrix is declared, but does not contain definite known values before it is used. When an uninitialized matrix is created,
#  whatever values were in the allocated memory at the time will appear as the initial values.

# Construct a 5x3 matrix, uninitialized:

x = torch.empty(5, 3)
print(x)