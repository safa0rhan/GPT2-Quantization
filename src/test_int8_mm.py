import torch
import int8_matmul

def int8_matmul_forward(A, B):
    return int8_matmul.int8_matmul_forward(A, B)

# Quick test
A = torch.randint(-128, 127, (4, 8), dtype=torch.int8, device='cuda')
B = torch.randint(-128, 127, (8, 10), dtype=torch.int8, device='cuda')

C = int8_matmul_forward(A, B)
print(C.shape)
print(C.dtype)
print(C.device)
print(C)