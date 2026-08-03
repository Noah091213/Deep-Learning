import torch

A = torch.rand(3,3)
B = torch.rand(3,3)

C = torch.matmul(A,B)

print("Matrix A:")
print(A)

print("\nMatrix B:")
print(B)

print("\nA x B:")
print(C)