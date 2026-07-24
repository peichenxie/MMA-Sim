import torch
from mmasim_kernels import MMAKernel


def get_random_tensor(m, n, dtype, device="cuda"):
    return torch.randn([m, n], device=device).to(dtype)


def is_independent(kernel: MMAKernel, trials: int = 100) -> bool:
    m, n, k = kernel.m, kernel.n, kernel.k
    for _ in range(trials):
        a = get_random_tensor(1, k, kernel.a_type)
        b = get_random_tensor(1, k, kernel.b_type)
        c = get_random_tensor(1, 1, kernel.c_type)
        A = a.expand(m, k)
        B = b.expand(n, k).T
        C = c.expand(m, n)
        D = kernel(A, B, C)
        is_diff = D != D[0, 0].item()
        if torch.any(is_diff):
            return False
    return True
