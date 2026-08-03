import torch

# Check for CUDA cores on the pc (Not needed for future programs, but a nice check)
print("Is cuda cores available? ", torch.cuda.is_available())

# Create the random 3x3 matrices
a = torch.rand(3,3)
b = torch.rand(3,3)

# Multiply a and b
c = torch.matmul(a, b)

# Print all 3
print("A = ", a)
print("B = ", b)
print("A*B = ", c)