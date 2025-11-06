import math
import torch
from torch import nn


def model_distance(a: nn.Module, b: nn.Module, device=torch.device("cuda")):
    l2_square = 0
    l1 = 0
    linf = 0
    num_params = 0
    for (name, param_a), (name, param_b) in zip(a.named_parameters(), b.named_parameters()):
        param_a, param_b = param_a.data.to(device), param_b.data.to(device)
        l2_square += ((param_a - param_b) ** 2).sum().item()
        l1 += (param_a.to(torch.float64) - param_b.to(torch.float64)).abs().sum().item()
        linf = max(linf, (param_a - param_b).abs().max().item())
        num_params += param_a.numel()
    avg = l1 / num_params
    l2 = math.sqrt(l2_square)
    print(f"l2: {l2}")
    print(f"l1: {l1}")
    print(f"linf: {linf}")
    print(f"avg: {avg}")
    print(f"l2 square: {l2_square}")
    print("-" * 30)
    return l2, l1, linf, avg, l2_square
