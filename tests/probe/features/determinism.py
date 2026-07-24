import torch
from mmasim_kernels import MMAKernel


def get_random_tensor(m, n, dtype, device="cuda"):
    return torch.randn([m, n], device=device).to(dtype)


def is_deterministic(kernel: MMAKernel, trials: int = 100, runs: int = 100) -> bool:
    m, n, k = kernel.m, kernel.n, kernel.k
    for _ in range(trials):
        A = get_random_tensor(m, k, kernel.a_type)
        B = get_random_tensor(n, k, kernel.b_type).T
        C = get_random_tensor(m, n, kernel.c_type)
        D = kernel(A, B, C)
        for _ in range(runs):
            res = kernel(A, B, C)
            is_diff = res.view(torch.int32) != D.view(torch.int32)
            if torch.any(is_diff):
                return False
    return True
